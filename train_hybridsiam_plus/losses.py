"""
losses.py
HybridSiam-CD+ — Loss Functions

Combined Focal + Dice loss to address the severe foreground/background
class imbalance characteristic of building change detection, where
changed pixels constitute only a small fraction of the total image
area (Lin et al., 2017; He and Garcia, 2009; Milletari, Navab and
Ahmadi, 2016).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss. Down-weights easy (well-classified) examples
    and up-weights the minority change class via alpha."""

    def __init__(self, alpha=0.75, gamma=2.5, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_term = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_term * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """Soft Dice loss — directly optimises the overlap between predicted
    and ground-truth change regions (Milletari, Navab and Ahmadi, 2016)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss used to train HybridSiam-CD+."""

    def __init__(
        self,
        focal_alpha=0.75,
        focal_gamma=2.5,
        focal_weight=1.0,
        dice_weight=1.0,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return total, {
            "focal": focal_loss.item(),
            "dice": dice_loss.item(),
            "total": total.item(),
        }
