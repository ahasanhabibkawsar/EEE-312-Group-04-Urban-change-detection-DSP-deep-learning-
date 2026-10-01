"""
Complete Siamese U-Net with ResNet34 backbone + Edge Head + BAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResNetEncoder
from .fusion import SiameseFeatureFusion
from .decoder import ChangeDecoder


class EdgeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(512, 64, 1)
        self.edge_predict = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, features, bottleneck):
        f0 = features[0]
        f1 = features[1]
        fused = self.conv_fuse(torch.cat([f0, f1], dim=1))
        context = self.context_conv(bottleneck)
        context = F.interpolate(context, size=fused.shape[-2:], mode='bilinear', align_corners=False)
        combined = fused + context
        edge_logits = self.edge_predict(combined)
        edge_logits = F.interpolate(edge_logits, scale_factor=4, mode='bilinear', align_corners=False)
        return edge_logits


class SiameseUNet(nn.Module):
    def __init__(self, in_channels=4, pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels, pretrained=pretrained)
        channels_list = self.encoder.channels  # [64, 64, 128, 256, 512]
        self.fusion = SiameseFeatureFusion(channels_list)
        self.decoder = ChangeDecoder()
        self.edge_head = EdgeHead()

    def forward(self, before, after, return_edges=False, return_distance=False):
        enc_out = self.encoder(before, after)

        if return_distance:
            diff = enc_out["before_bottleneck"] - enc_out["after_bottleneck"]
            dist = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-6)
            distance_map = F.interpolate(dist, size=before.shape[-2:], mode='bilinear', align_corners=False)
            return distance_map

        fusion_out = self.fusion(enc_out)
        # fusion_out has "fused_skips": list of 4 (stages 1-4) and "fused_bottleneck": stage4
        change_logits = self.decoder(fusion_out["fused_bottleneck"], fusion_out["fused_skips"])

        edge_logits = self.edge_head(enc_out["before_features"], enc_out["before_bottleneck"])

        if return_edges:
            return change_logits, edge_logits
        else:
            return change_logits