"""
Complete Siamese U-Net (v2):

    ResNet34 Siamese encoder (4-ch RGB + Canny, ImageNet init)
        ↓
    BAM spatial-temporal attention + Siam-Concat fusion at every scale
        ↓
    Edge head on the FUSED high-resolution features  ──┐ edge features
        ↓                                              │
    U-Net decoder (1/32 -> 1/2) + edge-guided fusion ←─┘
        ↓
    Full-resolution refinement -> change logits

Inference on large images
-------------------------
In eval mode, inputs larger than the training tile (256 x 256) are
predicted automatically with an overlapping sliding window whose tile
predictions are blended with a smooth weight window. This keeps the
network at the scale it was trained on (native 0.5 m / pixel) and lets
every script pass a full 1024 x 1024 LEVIR-CD image directly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IMAGE_SIZE, INFERENCE_TILE_OVERLAP, INFERENCE_TILE_BATCH

from .encoder import ResNetEncoder
from .fusion import SiameseFeatureFusion
from .decoder import ChangeDecoder, conv_bn_relu


MODEL_VERSION = 2
_DIVISOR = 32   # total down-sampling of the encoder


class EdgeHead(nn.Module):
    """
    Auxiliary change-boundary head.

    Input: FUSED (bi-temporal) stage-0 and stage-1 features plus the
    fused bottleneck as global context. v1 used only the BEFORE image's
    features, which cannot contain the outlines of buildings that only
    exist in the AFTER image.

    Output: edge logits (full resolution) and a 32-channel edge feature
    map (1/2 resolution) that guides the change decoder.
    """

    def __init__(self, low_channels=(64, 64), context_channels=512, edge_channels=32):
        super().__init__()

        self.conv_fuse = conv_bn_relu(sum(low_channels), edge_channels)
        self.context_conv = nn.Conv2d(context_channels, edge_channels, kernel_size=1)
        self.edge_conv = conv_bn_relu(edge_channels, edge_channels)
        self.edge_predict = nn.Conv2d(edge_channels, 1, kernel_size=1)

    def forward(self, f0, f1, bottleneck, output_size):
        f1 = F.interpolate(f1, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv_fuse(torch.cat([f0, f1], dim=1))

        context = self.context_conv(bottleneck)
        context = F.interpolate(context, size=x.shape[-2:], mode="bilinear", align_corners=False)

        edge_features = self.edge_conv(x + context)

        edge_logits = self.edge_predict(edge_features)
        edge_logits = F.interpolate(edge_logits, size=output_size, mode="bilinear", align_corners=False)

        return edge_logits, edge_features


class SiameseUNet(nn.Module):

    def __init__(
        self,
        in_channels=4,
        pretrained=False,
        tile_size=IMAGE_SIZE,
        tile_overlap=INFERENCE_TILE_OVERLAP,
        tile_batch=INFERENCE_TILE_BATCH,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_batch = tile_batch

        self.encoder = ResNetEncoder(in_channels=in_channels, pretrained=pretrained)
        channels = self.encoder.channels                      # [64, 64, 128, 256, 512]

        self.fusion = SiameseFeatureFusion(channels)
        self.edge_head = EdgeHead(low_channels=(channels[0], channels[1]),
                                  context_channels=channels[4])
        self.decoder = ChangeDecoder(encoder_channels=channels)

    # =====================================================
    # Core forward on one tile (H, W divisible by 32)
    # =====================================================

    def _forward_core(self, before, after, return_edges=False):
        output_size = before.shape[-2:]

        enc_out = self.encoder(before, after)
        fusion_out = self.fusion(enc_out)

        s0, s1 = fusion_out["fused_skips"][:2]
        edge_logits, edge_features = self.edge_head(
            s0, s1, fusion_out["fused_bottleneck"], output_size
        )

        change_logits = self.decoder(
            fusion_out["fused_bottleneck"],
            fusion_out["fused_skips"],
            edge_features=edge_features,
            output_size=output_size,
        )

        if return_edges:
            return change_logits, edge_logits
        return change_logits

    def _forward_padded(self, before, after, return_edges=False):
        """Pad H, W up to a multiple of 32, run, crop back."""

        H, W = before.shape[-2:]
        pad_h = (-H) % _DIVISOR
        pad_w = (-W) % _DIVISOR

        if pad_h or pad_w:
            before = _pad(before, pad_h, pad_w)
            after = _pad(after, pad_h, pad_w)

        out = self._forward_core(before, after, return_edges)

        if pad_h or pad_w:
            if return_edges:
                return out[0][..., :H, :W], out[1][..., :H, :W]
            return out[..., :H, :W]
        return out

    # =====================================================
    # Public forward
    # =====================================================

    def forward(self, before, after, return_edges=False, return_distance=False):

        if return_distance:
            # Feature distance map for the (optional) BCL contrastive loss
            enc_out = self.encoder(before, after)
            diff = enc_out["before_bottleneck"] - enc_out["after_bottleneck"]
            dist = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-6)
            return F.interpolate(dist, size=before.shape[-2:], mode="bilinear", align_corners=False)

        H, W = before.shape[-2:]

        if (not self.training and self.tile_size
                and (H > self.tile_size or W > self.tile_size)):
            return self.sliding_window_forward(before, after, return_edges=return_edges)

        return self._forward_padded(before, after, return_edges)

    # =====================================================
    # Sliding-window inference for large images
    # =====================================================

    @torch.no_grad()
    def sliding_window_forward(self, before, after, return_edges=False,
                               tile_size=None, overlap=None, tile_batch=None):
        """
        Predict an arbitrarily large image pair with overlapping tiles.

        Tile logits are blended with a separable weight window that is
        1 in the tile centre and ramps down towards the tile border
        (inside the overlap), so seams between tiles are invisible and
        border pixels - which see less context - count less.
        """

        tile = tile_size or self.tile_size
        overlap = self.tile_overlap if overlap is None else overlap
        overlap = min(max(int(overlap), 0), tile // 2)
        tile_batch = tile_batch or self.tile_batch

        B, _, H, W = before.shape

        # Images smaller than one tile are padded up to a full tile
        pad_h = max(tile - H, 0)
        pad_w = max(tile - W, 0)
        if pad_h or pad_w:
            before = _pad(before, pad_h, pad_w)
            after = _pad(after, pad_h, pad_w)

        Hp, Wp = before.shape[-2:]
        stride = tile - overlap

        ys = _tile_starts(Hp, tile, stride)
        xs = _tile_starts(Wp, tile, stride)
        positions = [(y, x) for y in ys for x in xs]

        weight = _blend_window(tile, overlap, before.device, before.dtype)

        change_acc = before.new_zeros(B, 1, Hp, Wp)
        edge_acc = before.new_zeros(B, 1, Hp, Wp) if return_edges else None
        weight_acc = before.new_zeros(1, 1, Hp, Wp)

        per_pass = max(1, tile_batch // B)

        for i in range(0, len(positions), per_pass):
            chunk = positions[i:i + per_pass]

            b_tiles = torch.cat([before[..., y:y + tile, x:x + tile] for y, x in chunk], dim=0)
            a_tiles = torch.cat([after[..., y:y + tile, x:x + tile] for y, x in chunk], dim=0)

            out = self._forward_padded(b_tiles, a_tiles, return_edges=return_edges)
            change_out, edge_out = out if return_edges else (out, None)

            for j, (y, x) in enumerate(chunk):
                sl = slice(j * B, (j + 1) * B)
                change_acc[..., y:y + tile, x:x + tile] += change_out[sl] * weight
                if return_edges:
                    edge_acc[..., y:y + tile, x:x + tile] += edge_out[sl] * weight
                weight_acc[..., y:y + tile, x:x + tile] += weight

        change_logits = (change_acc / weight_acc)[..., :H, :W]

        if return_edges:
            return change_logits, (edge_acc / weight_acc)[..., :H, :W]
        return change_logits


# =========================================================
# Helpers
# =========================================================

def _pad(x, pad_h, pad_w):
    """Reflect-pad bottom / right (replicate if the image is too small)."""

    H, W = x.shape[-2:]
    mode = "reflect" if (pad_h < H and pad_w < W) else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)


def _tile_starts(length, tile, stride):
    """Start offsets so that tiles cover [0, length) exactly."""

    if length <= tile:
        return [0]

    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def _blend_window(tile, overlap, device, dtype):
    """Separable 2-D window: cosine ramp over `overlap` px at each border."""

    w = torch.ones(tile, device=device, dtype=dtype)

    if overlap > 0:
        ramp = torch.arange(overlap, device=device, dtype=dtype)
        ramp = 0.5 - 0.5 * torch.cos(math.pi * (ramp + 1) / (overlap + 1))   # (0, 1)
        w[:overlap] = ramp
        w[-overlap:] = ramp.flip(0)

    return (w[:, None] * w[None, :]).view(1, 1, tile, tile)


@torch.no_grad()
def predict_probabilities(model, before, after, tta=False):
    """
    Change probabilities for a batch, with optional test-time augmentation.

    TTA averages the probabilities of 4 geometric views
    (identity, horizontal flip, vertical flip, 180° rotation);
    each view is un-flipped before averaging.
    """

    model.eval()

    if not tta:
        return torch.sigmoid(model(before, after))

    views = [(), (3,), (2,), (2, 3)]
    total = 0.0

    for dims in views:
        b = torch.flip(before, dims) if dims else before
        a = torch.flip(after, dims) if dims else after
        p = torch.sigmoid(model(b, a))
        total = total + (torch.flip(p, dims) if dims else p)

    return total / len(views)
