"""
Loss functions for urban change detection.

Includes:
    - Dice Loss
    - BCE + Dice Combined Loss
    - Focal Loss
    - BCE + Dice + Focal Combined Loss
    - Class-Balanced Edge Loss (EGCTNet / HED)
    - Batch-Balanced Contrastive Loss (STANet)
    - MultiTaskChangeLoss  (the loss actually used by train.py)

Training objective (v2)
-----------------------

    L = w_bce · BCE(change) + w_dice · Dice(change) + w_edge · CB-BCE(edge)

    BCE   : pixel-wise accuracy of the change map
    Dice  : directly optimises the overlap (F1 / IoU) and is insensitive
            to the ~95 % unchanged background
    CB-BCE: class-balanced BCE on the change-boundary map. Boundary
            pixels are only ~1-2 % of a mask; plain BCE (used by v1)
            lets the edge head predict "no edge" everywhere. Each class
            is re-weighted by the frequency of the other class.

The edge target is computed on the fly with a Sobel operator applied
to the ground-truth mask (see make_edge_target).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing.image_processing import sobel_edge_detection


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    per_sample=False (default) pools all pixels of the batch, which is
    stable when many crops contain no change at all.
    """

    def __init__(self, smooth=1.0, per_sample=False):
        super().__init__()
        self.smooth = smooth
        self.per_sample = per_sample

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)

        if self.per_sample:
            probabilities = probabilities.reshape(probabilities.shape[0], -1)
            targets = targets.reshape(targets.shape[0], -1)
            dims = 1
        else:
            probabilities = probabilities.reshape(-1)
            targets = targets.reshape(-1)
            dims = 0

        intersection = (probabilities * targets).sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (
            probabilities.sum(dim=dims) + targets.sum(dim=dims) + self.smooth
        )
        return (1.0 - dice).mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return (self.bce_weight * self.bce(logits, targets)
                + self.dice_weight * self.dice(logits, targets))


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = ((self.alpha * targets + (1 - self.alpha) * (1 - targets))
                        * (1 - p_t) ** self.gamma)
        return (focal_weight * ce_loss).mean()


class CombinedLoss(nn.Module):
    """
    BCE + Dice + Focal.
    """

    def __init__(self, bce_w=0.5, dice_w=0.25, focal_w=0.25):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.focal_w = focal_w

    def forward(self, logits, targets):
        return (self.bce_w * self.bce(logits, targets)
                + self.dice_w * self.dice(logits, targets)
                + self.focal_w * self.focal(logits, targets))


class ClassBalancedEdgeLoss(nn.Module):
    """
    Class-balanced binary cross-entropy for edge detection
    (HED, Xie & Tu 2015; used for the edge branch of EGCTNet).

        beta = |non-edge| / |all|
        L    = mean( beta · y · BCE  +  (1 - beta) · (1 - y) · BCE )

    Rare edge pixels get weight beta (~0.98), common background pixels
    get 1 - beta (~0.02).
    """

    def __init__(self):
        super().__init__()

    def forward(self, edge_logits, edge_targets):
        edge_targets = edge_targets.float()

        num_total = edge_targets.numel()
        num_edges = edge_targets.sum()

        if num_edges.item() == 0:
            # No boundary in this batch: plain BCE keeps the head calibrated
            return F.binary_cross_entropy_with_logits(edge_logits, edge_targets) * 0.1

        beta = (num_total - num_edges) / num_total
        weights = beta * edge_targets + (1.0 - beta) * (1.0 - edge_targets)

        return F.binary_cross_entropy_with_logits(
            edge_logits, edge_targets, weight=weights
        )


class BatchBalancedContrastiveLoss(nn.Module):
    """
    Batch-Balanced Contrastive Loss (BCL) from STANet.

    L = (1/(2 n_u)) Σ_unchanged D²  +  (1/(2 n_c)) Σ_changed max(0, m - D)²

    where D is the L2 distance between Before / After features.
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance_map, targets):
        dist_flat = distance_map.reshape(-1)
        target_flat = targets.reshape(-1)

        n_u = (target_flat == 0).sum().float()
        n_c = (target_flat == 1).sum().float()

        zero = distance_map.new_zeros(())

        loss_u = ((dist_flat ** 2) * (1 - target_flat)).sum() / (2.0 * n_u) if n_u > 0 else zero

        if n_c > 0:
            hinge = torch.clamp(self.margin - dist_flat, min=0.0) ** 2
            loss_c = (hinge * target_flat).sum() / (2.0 * n_c)
        else:
            loss_c = zero

        return loss_u + loss_c


# =========================================================
# Multi-task loss used for training
# =========================================================

def make_edge_target(mask):
    """
    Change-boundary target: Sobel gradient magnitude of the binary
    ground-truth mask, thresholded at 0.5.
    """

    return sobel_edge_detection(mask.float())


class MultiTaskChangeLoss(nn.Module):
    """
    L = w_bce · BCE + w_dice · Dice + w_edge · ClassBalancedEdgeLoss
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, edge_weight=0.4):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight

        self.dice = DiceLoss()
        self.edge = ClassBalancedEdgeLoss()

    def forward(self, change_logits, mask, edge_logits=None):
        """
        Returns
        -------
        total : Tensor (scalar)
        parts : dict of floats {"bce", "dice", "edge"}
        """

        bce = F.binary_cross_entropy_with_logits(change_logits, mask)
        dice = self.dice(change_logits, mask)
        total = self.bce_weight * bce + self.dice_weight * dice

        parts = {"bce": bce.item(), "dice": dice.item(), "edge": 0.0}

        if edge_logits is not None and self.edge_weight > 0:
            edge_target = make_edge_target(mask)
            edge = self.edge(edge_logits, edge_target)
            total = total + self.edge_weight * edge
            parts["edge"] = edge.item()

        return total, parts
