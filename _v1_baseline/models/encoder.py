"""
ResNet34-based Siamese Encoder with 4‑channel input support.
Pre‑trained on ImageNet, first conv adapted to accept 4 channels.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """
    ResNet34 encoder that outputs multi‑scale features.
    Adapted for 4‑channel input (RGB + Canny edge).
    """

    def __init__(self, in_channels=4, pretrained=True):
        super().__init__()
        # Load pre‑trained ResNet34
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Modify the first convolution to accept in_channels (default 4)
        old_conv = resnet.conv1  # Shape: [64, 3, 7, 7]
        new_conv = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        # Initialise the new conv with the pre‑trained weights
        with torch.no_grad():
            # Average over the RGB channels
            new_conv.weight[:, :3] = old_conv.weight
            # Copy the average to the extra channel(s)
            if in_channels > 3:
                new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv

        # Extract stages
        self.stage0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )  # 1/4 resolution, 64 channels

        self.stage1 = resnet.layer1  # 1/4, 64 channels
        self.stage2 = resnet.layer2  # 1/8, 128 channels
        self.stage3 = resnet.layer3  # 1/16, 256 channels
        self.stage4 = resnet.layer4  # 1/32, 512 channels

        # Store channel numbers for easy access
        self.channels = [64, 64, 128, 256, 512]  # stage0,1,2,3,4

    def encode(self, x):
        """
        Encode a single image into multi‑scale features.
        Returns:
            features: list of 5 feature maps [stage0, stage1, stage2, stage3, stage4]
            bottleneck: the deepest feature map (stage4)
        """
        s0 = self.stage0(x)   # 1/4
        s1 = self.stage1(s0)  # 1/4
        s2 = self.stage2(s1)  # 1/8
        s3 = self.stage3(s2)  # 1/16
        s4 = self.stage4(s3)  # 1/32

        features = [s0, s1, s2, s3, s4]
        bottleneck = s4
        return features, bottleneck

    def forward(self, before, after):
        """
        Encode both images with shared weights.
        Returns a dict with before/after features and bottlenecks.
        """
        before_features, before_bottleneck = self.encode(before)
        after_features, after_bottleneck = self.encode(after)

        return {
            "before_features": before_features,
            "after_features": after_features,
            "before_bottleneck": before_bottleneck,
            "after_bottleneck": after_bottleneck,
        }