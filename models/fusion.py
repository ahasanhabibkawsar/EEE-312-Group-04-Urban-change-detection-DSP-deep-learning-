"""
Feature fusion: Spatial-Temporal Attention (BAM) + Siam-Concat.

For every encoder scale s the Before / After features are fused as

    F_b', F_a'  = BAM(F_b, F_a)                       (stages 1-4)
    D           = | F_b' - F_a' |
    F_fused     = ReLU(BN(Conv1x1( [F_b', F_a', D] )))   3C -> C

BAM (Basic spatial-temporal Attention Module, STANet, Chen & Shi 2020)
----------------------------------------------------------------------
The Before and After feature maps are placed side by side
([B, C, H, 2W]) so that every pixel of either date can attend to every
pixel of BOTH dates:

    Q = Wq X,  K = Wk X̃,  V = Wv X̃         (X̃ = X average-pooled by r)
    A = softmax( Qᵀ K / sqrt(d_k) )         [2HW x 2HW/r²]
    X' = X + γ · V Aᵀ                        (γ initialised to 0)

v2 fixes compared with v1
-------------------------
* v1 stacked the dates as [B, C, H, W, 2] -> permute -> view(2B, ...),
  which interleaves them as [b0_before, b0_after, b1_before, ...], but
  then split the result as out[:B] / out[B:]. With batch size 4 the
  "before" branch of sample 1 received the attention output of sample
  0's AFTER image, etc. Predictions depended on which images shared a
  batch. Fixed by never mixing the batch axis.
* v1 attended within each image only (spatial attention). The dates
  are now concatenated along the width, which is the actual
  spatial-TEMPORAL attention of STANet.
* Keys / values are average-pooled at high-resolution stages
  (spatial-reduction attention), cutting the attention matrix of stage 1
  from 4096 x 4096 per image to 8192 x 512 per pair - affordable on a
  laptop GPU.
* γ = 0 at initialisation, so BAM starts as the identity and cannot
  disturb the ImageNet-pre-trained features early in training.
* The stage-0 fusion that v1 computed and then discarded is now used as
  the 1/2-resolution skip connection (without BAM, too large).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BAM(nn.Module):
    """
    Basic Spatial-Temporal Attention Module (STANet).
    """

    def __init__(self, in_channels, key_channels=None, pool_size=1):
        super().__init__()

        key_channels = key_channels or max(in_channels // 8, 8)

        self.key_channels = key_channels
        self.pool_size = pool_size

        self.q_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable residual scale, 0 at start -> identity mapping
        self.gamma = nn.Parameter(torch.zeros(1))

    def _pool(self, x):
        if self.pool_size > 1:
            return F.avg_pool2d(x, self.pool_size, ceil_mode=True)
        return x

    def forward(self, before, after):
        B, C, H, W = before.shape

        # Both dates side by side:  [B, C, H, 2W]
        x = torch.cat([before, after], dim=3)

        # Keys / values from the (optionally pooled) dates, also side by side
        x_kv = torch.cat([self._pool(before), self._pool(after)], dim=3)

        q = self.q_conv(x).flatten(2)          # [B, d, N]    N = 2HW
        k = self.k_conv(x_kv).flatten(2)       # [B, d, M]    M = 2HW / r²
        v = self.v_conv(x_kv).flatten(2)       # [B, C, M]

        attention = torch.bmm(q.transpose(1, 2), k) * (self.key_channels ** -0.5)
        attention = attention.softmax(dim=-1)  # [B, N, M]

        out = torch.bmm(v, attention.transpose(1, 2))   # [B, C, N]
        out = out.view(B, C, H, 2 * W)

        out = x + self.gamma * out

        return out[..., :W], out[..., W:]


class SiameseFeatureFusion(nn.Module):
    """
    BAM + Siam-Concat fusion at every encoder scale.
    """

    def __init__(
        self,
        channels_list,
        bam_stages=(1, 2, 3, 4),
        bam_pool_sizes=(1, 4, 2, 1, 1),
    ):
        super().__init__()

        # channels_list: [64, 64, 128, 256, 512]
        self.bam_stages = tuple(bam_stages)

        self.bams = nn.ModuleDict({
            str(i): BAM(c, pool_size=bam_pool_sizes[i])
            for i, c in enumerate(channels_list)
            if i in self.bam_stages
        })

        # 1x1 convolutions compress [before, after, |diff|] (3C) back to C
        self.reduce_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 3, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
            for c in channels_list
        ])

    def forward(self, encoder_output):
        before_features = encoder_output["before_features"]
        after_features = encoder_output["after_features"]

        fused_all = []

        for i, (bf, af) in enumerate(zip(before_features, after_features)):

            # 1. Spatial-temporal attention
            if str(i) in self.bams:
                bf, af = self.bams[str(i)](bf, af)

            # 2. Absolute difference
            difference = torch.abs(bf - af)

            # 3. Siam-Concat + 1x1 compression
            fused = self.reduce_convs[i](torch.cat([bf, af, difference], dim=1))

            fused_all.append(fused)

        return {
            "fused_skips": fused_all[:4],       # stages 0-3 (1/2 ... 1/16)
            "fused_bottleneck": fused_all[4],   # stage 4    (1/32)
        }
