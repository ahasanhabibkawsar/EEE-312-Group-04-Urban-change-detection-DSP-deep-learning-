"""
Loss functions for urban change detection.

Includes:
    - Dice Loss
    - BCE Loss
    - Focal Loss
    - BCE + Dice Combined Loss
    - BCE + Dice + Focal Combined Loss
    - Class-Balanced Edge Loss (EGCTNet)
    - Batch-Balanced Contrastive Loss (STANet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.reshape(probabilities.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        intersection = (probabilities * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probabilities.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        loss = 1.0 - dice
        return loss.mean()


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
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


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
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


class CombinedLoss(nn.Module):
    """
    Powerful combined loss: BCE + Dice + Focal.
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
        return (self.bce_w * self.bce(logits, targets) +
                self.dice_w * self.dice(logits, targets) +
                self.focal_w * self.focal(logits, targets))


class ClassBalancedEdgeLoss(nn.Module):
    """
    Class-Balanced Cross Entropy Loss for Edge Detection (EGCTNet).
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, edge_logits, edge_targets):
        edge_targets_flat = edge_targets.view(-1)
        edge_logits_flat = edge_logits.view(-1)
        
        num_edges = edge_targets_flat.sum()
        num_non_edges = edge_targets_flat.numel() - num_edges
        
        if num_edges == 0:
            return (edge_logits_flat * 0).mean()
        
        beta = num_non_edges.float() / edge_targets_flat.numel()
        losses = self.bce(edge_logits_flat, edge_targets_flat)
        weights = beta * edge_targets_flat + (1 - beta) * (1 - edge_targets_flat)
        weighted_loss = (weights * losses).mean()
        
        return weighted_loss


class BatchBalancedContrastiveLoss(nn.Module):
    """
    Batch-Balanced Contrastive Loss (BCL) from STANet.
    
    L = (1/(2*n_u)) * sum(no_change) (D^2) 
        + (1/(2*n_c)) * sum(change) max(0, m - D)^2
    
    Where D is the L2 distance between before/after features.
    This dynamically balances the contribution of changed and unchanged pixels.
    """
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance_map, targets):
        """
        Parameters
        ----------
        distance_map : Tensor
            L2 distance between features. Shape: [B, 1, H, W]
        targets : Tensor
            Ground truth mask. Shape: [B, 1, H, W]
        """
        dist_flat = distance_map.view(-1)
        target_flat = targets.view(-1)
        
        # Number of no-change and change pixels
        n_u = (target_flat == 0).sum().float()
        n_c = (target_flat == 1).sum().float()
        
        # No-change loss: pull close (minimize distance squared)
        if n_u > 0:
            loss_u = (dist_flat * (1 - target_flat)).sum() / (2.0 * n_u)
        else:
            loss_u = torch.tensor(0.0, device=distance_map.device)
            
        # Change loss: push apart (max(0, margin - distance)^2)
        if n_c > 0:
            hinge_loss = torch.clamp(self.margin - dist_flat, min=0.0) ** 2
            loss_c = (hinge_loss * target_flat).sum() / (2.0 * n_c)
        else:
            loss_c = torch.tensor(0.0, device=distance_map.device)
            
        return loss_u + loss_c