"""
U-Net decoder for the fused ResNet34 features.

    bottleneck (512, 1/32)
        ↓ up ×2 + skip3 (256, 1/16)  -> DecoderBlock -> 256
        ↓ up ×2 + skip2 (128, 1/8)   -> DecoderBlock -> 128
        ↓ up ×2 + skip1 ( 64, 1/4)   -> DecoderBlock ->  64
        ↓ up ×2 + skip0 ( 64, 1/2)   -> DecoderBlock ->  32
        ⊕ edge features (32, 1/2)     -> edge-guided fusion
        ↓ up ×2 (full resolution)     -> 3x3 refinement conv -> 1x1 -> logits

v2 fixes compared with v1
-------------------------
* v1 concatenated the bottleneck with the SAME stage-4 tensor
  (fused_bottleneck == fused_skips[-1]) at 1/32 resolution, i.e. its
  first "up-sampling" block did not up-sample and duplicated features.
  Each block now up-samples by 2 and merges the next shallower skip.
* v1 stopped at 1/4 resolution and up-sampled the logits ×4 bilinearly,
  which blurs building outlines ("halos"). The decoder now goes down to
  1/2 resolution using the stage-0 skip, and the final ×2 up-sampling is
  followed by a learnable full-resolution refinement convolution.
* The auxiliary edge branch now GUIDES the change prediction: its
  features are concatenated into the last decoder stage (edge-guided
  fusion, as in EGCTNet) instead of being a dead-end head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    """
    Up-sample to the skip resolution, concatenate the skip, two 3x3 convs.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            conv_bn_relu(in_channels + skip_channels, out_channels),
            conv_bn_relu(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ChangeDecoder(nn.Module):
    """
    Decoder that outputs change logits at full input resolution.
    """

    def __init__(
        self,
        encoder_channels=(64, 64, 128, 256, 512),
        decoder_channels=(256, 128, 64, 32),
        edge_channels=32,
    ):
        super().__init__()

        c0, c1, c2, c3, c4 = encoder_channels
        d3, d2, d1, d0 = decoder_channels

        self.decoder3 = DecoderBlock(c4, c3, d3)   # 1/32 -> 1/16
        self.decoder2 = DecoderBlock(d3, c2, d2)   # 1/16 -> 1/8
        self.decoder1 = DecoderBlock(d2, c1, d1)   # 1/8  -> 1/4
        self.decoder0 = DecoderBlock(d1, c0, d0)   # 1/4  -> 1/2

        self.edge_channels = edge_channels
        self.edge_fusion = conv_bn_relu(d0 + edge_channels, d0)

        # Full-resolution refinement after the final ×2 up-sampling
        self.refine = conv_bn_relu(d0, 16)
        self.change_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, fused_bottleneck, fused_skips, edge_features=None, output_size=None):
        """
        fused_bottleneck : [B, 512, H/32, W/32]
        fused_skips      : [s0 (1/2), s1 (1/4), s2 (1/8), s3 (1/16)]
        edge_features    : [B, edge_channels, H/2, W/2] or None
        output_size      : (H, W); defaults to 2 x the stage-0 size
        """

        s0, s1, s2, s3 = fused_skips

        x = self.decoder3(fused_bottleneck, s3)
        x = self.decoder2(x, s2)
        x = self.decoder1(x, s1)
        x = self.decoder0(x, s0)                                   # 1/2

        if edge_features is None:
            edge_features = x.new_zeros(x.shape[0], self.edge_channels, *x.shape[-2:])
        x = self.edge_fusion(torch.cat([x, edge_features], dim=1))

        if output_size is None:
            output_size = (s0.shape[-2] * 2, s0.shape[-1] * 2)

        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        x = self.refine(x)

        return self.change_head(x)
