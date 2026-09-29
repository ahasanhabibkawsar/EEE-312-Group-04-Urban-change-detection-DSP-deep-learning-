"""
Loss functions for Urban Change Detection.
"""

from .losses import (
    DiceLoss,
    BCEDiceLoss,
    FocalLoss,
    CombinedLoss,
    ClassBalancedEdgeLoss,
    BatchBalancedContrastiveLoss,
    MultiTaskChangeLoss,
    make_edge_target,
)
