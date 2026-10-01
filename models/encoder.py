"""
ResNet34 Siamese encoder with 4-channel (RGB + Canny) input.

Pre-trained on ImageNet; the first convolution is widened from 3 to 4
input channels without destroying the pre-trained RGB filters:

    W_new[:, 0:3] = W_imagenet                    (RGB filters copied)
    W_new[:, 3]   = mean_c W_imagenet[:, c]       (edge filter = mean RGB filter)

Because the edge channel is standardised like the RGB channels
(see preprocessing/dsp_processing.py), the edge filter initially
responds to edges the way a luminance filter would.

Output scales for a 256 x 256 input
-----------------------------------
    stage0  conv1 + bn + relu         64 ch   1/2   (128 x 128)
    stage1  maxpool + layer1          64 ch   1/4   ( 64 x  64)
    stage2  layer2                   128 ch   1/8   ( 32 x  32)
    stage3  layer3                   256 ch   1/16  ( 16 x  16)
    stage4  layer4                   512 ch   1/32  (  8 x   8)

v2 change: stage0 is now taken BEFORE the max-pool (1/2 resolution),
giving the decoder a genuinely high-resolution skip connection for
sharper building boundaries and small changes.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """
    Weight-sharing ResNet34 encoder that returns multi-scale features.
    """

    def __init__(self, in_channels=4, pretrained=False):
        super().__init__()

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)

        if in_channels != 3:
            old_conv = resnet.conv1                     # [64, 3, 7, 7]
            new_conv = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False,
            )

            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if in_channels > 3:
                    mean_filter = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:] = mean_filter.repeat(1, in_channels - 3, 1, 1)

            resnet.conv1 = new_conv

        self.in_channels = in_channels

        self.stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # 1/2
        self.pool = resnet.maxpool
        self.stage1 = resnet.layer1                                          # 1/4
        self.stage2 = resnet.layer2                                          # 1/8
        self.stage3 = resnet.layer3                                          # 1/16
        self.stage4 = resnet.layer4                                          # 1/32

        self.channels = [64, 64, 128, 256, 512]

    def encode(self, x):
        """
        Encode a batch of images into multi-scale features.

        Returns
        -------
        features   : list of 5 tensors [stage0 ... stage4]
        bottleneck : stage4 tensor
        """

        s0 = self.stage0(x)
        s1 = self.stage1(self.pool(s0))
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        return [s0, s1, s2, s3, s4], s4

    def forward(self, before, after):
        """
        Encode both dates with the SAME weights.

        The two dates are processed as one concatenated batch
        (one pass through the network, identical weights), then split.
        """

        batch = before.shape[0]
        features, _ = self.encode(torch.cat([before, after], dim=0))

        before_features = [f[:batch] for f in features]
        after_features = [f[batch:] for f in features]

        return {
            "before_features": before_features,
            "after_features": after_features,
            "before_bottleneck": before_features[-1],
            "after_bottleneck": after_features[-1],
        }
