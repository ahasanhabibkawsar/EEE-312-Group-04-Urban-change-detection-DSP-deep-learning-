"""
U-Net Decoder for ResNet34 features.
Outputs at full resolution (1/1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """
    One decoder block: upsample + concat skip + two convs.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ChangeDecoder(nn.Module):
    """
    Decoder for ResNet34 features.
    Outputs change logits at full resolution (1/1).
    """
    def __init__(self):
        super().__init__()
        # Decoder4: bottleneck (512) + skip stage4 (512) -> 256
        self.decoder4 = DecoderBlock(512, 512, 256)
        # Decoder3: 256 + skip stage3 (256) -> 128
        self.decoder3 = DecoderBlock(256, 256, 128)
        # Decoder2: 128 + skip stage2 (128) -> 64
        self.decoder2 = DecoderBlock(128, 128, 64)
        # Decoder1: 64 + skip stage1 (64) -> 64
        self.decoder1 = DecoderBlock(64, 64, 64)

        # Final change head (at 1/4 resolution)
        self.change_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, fused_bottleneck, fused_skips):
        """
        fused_bottleneck: [B, 512, H/32, W/32]
        fused_skips: list of 4 tensors [stage1, stage2, stage3, stage4]
                     stage1: [B, 64, H/4, W/4]
                     stage2: [B, 128, H/8, W/8]
                     stage3: [B, 256, H/16, W/16]
                     stage4: [B, 512, H/32, W/32]
        """
        s1, s2, s3, s4 = fused_skips
        x = self.decoder4(fused_bottleneck, s4)   # 1/16? Actually s4 is 1/32? Wait: s4 from fused features is stage4, which is 1/32. But our decoder4 upsamples to s4's size, so we go from 1/32 to 1/32? Let's trace:
        # fused_bottleneck is 1/32. decoder4 upsamples to s4 (1/32) -> stays 1/32, then conv -> 1/32.
        # Then decoder3 upsamples to s3 (1/16) -> 1/16.
        # decoder2 upsamples to s2 (1/8) -> 1/8.
        # decoder1 upsamples to s1 (1/4) -> 1/4.
        x = self.decoder3(x, s3)   # 1/16
        x = self.decoder2(x, s2)   # 1/8
        x = self.decoder1(x, s1)   # 1/4

        logits = self.change_head(x)  # [B, 1, H/4, W/4]

        # Upsample to full resolution (1/1)
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        return logits