"""
Feature Fusion Module with Spatial-Temporal Attention (BAM) and Siam-Concat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BAM(nn.Module):
    """
    Basic Spatial-Temporal Attention Module (BAM) from STANet.
    """
    def __init__(self, in_channels, key_channels=None):
        super().__init__()
        if key_channels is None:
            key_channels = in_channels // 8
        self.key_channels = key_channels
        
        self.q_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, before, after):
        B, C, H, W = before.shape
        
        # Stack along temporal dimension: [B, C, H, W, 2]
        x = torch.stack([before, after], dim=-1)
        
        # Reshape to [B*2, C, H, W]
        x_2 = x.permute(0, 4, 1, 2, 3).contiguous().view(B*2, C, H, W)
        
        # Compute Q, K, V
        Q = self.q_conv(x_2)  # [B*2, key_channels, H, W]
        K = self.k_conv(x_2)
        V = self.v_conv(x_2)
        
        # Reshape to matrices
        Q_flat = Q.view(B*2, self.key_channels, -1)
        K_flat = K.view(B*2, self.key_channels, -1)
        V_flat = V.view(B*2, C, -1)
        
        # Attention
        scale = self.key_channels ** -0.5
        attention = torch.bmm(K_flat.transpose(1, 2), Q_flat) * scale
        attention = self.softmax(attention)
        
        # Output
        out_flat = torch.bmm(V_flat, attention)
        out = out_flat.view(B*2, C, H, W)
        
        # Split back
        out_before = out[0:B]
        out_after = out[B:B*2]
        
        out_before = self.out_conv(out_before)
        out_after = self.out_conv(out_after)
        
        return before + out_before, after + out_after


class SiameseFeatureFusion(nn.Module):
    """
    Advanced Feature Fusion using BAM and Siam-Concat.
    (Before + After + Absolute Difference)
    """
    def __init__(self, channels_list):
        super().__init__()
        # channels_list: [64, 64, 128, 256, 512]
        self.bams = nn.ModuleList([BAM(c) for c in channels_list])
        
        # 1x1 Convolutions to compress the concatenated (Before + After + Diff) features
        # from 3*C channels back down to C channels so the decoder can accept them.
        self.reduce_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 3, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ) for c in channels_list
        ])
        
    def forward(self, encoder_output):
        before_features = encoder_output["before_features"]
        after_features = encoder_output["after_features"]
        before_bottleneck = encoder_output["before_bottleneck"]
        after_bottleneck = encoder_output["after_bottleneck"]
        
        fused_all = []
        for i, bam in enumerate(self.bams):
            if i < len(before_features):
                bf = before_features[i]
                af = after_features[i]
            else:
                bf = before_bottleneck
                af = after_bottleneck
                
            # 1. Apply Spatial-Temporal Attention
            fused_b, fused_a = bam(bf, af)
            
            # 2. Calculate absolute difference
            difference = torch.abs(fused_b - fused_a)
            
            # 3. THE UPGRADE: Concatenate all three features together!
            concat_features = torch.cat([fused_b, fused_a, difference], dim=1)
            
            # 4. Compress back to the original channel size
            reduced_features = self.reduce_convs[i](concat_features)
            
            fused_all.append(reduced_features)
        
        fused_skips = fused_all[1:5]  # stages 1, 2, 3, 4
        fused_bottleneck = fused_all[4]  # stage 4
        
        return {
            "fused_skips": fused_skips,
            "fused_bottleneck": fused_bottleneck,
        }