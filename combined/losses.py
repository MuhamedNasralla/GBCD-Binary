"""
losses.py
HybridSiam-CD+ (Combined) — Loss Function

Combined Focal + Dice loss to address the severe foreground/background
class imbalance characteristic of building change detection across all
four training domains (Lin et al., 2017; He and Garcia, 2009; Milletari,
Navab and Ahmadi, 2016). Matches the exact loss implementation used in
the combined multi-domain training notebook, including NaN-guard
fallback for numerically unstable batches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class FocalDiceLoss(nn.Module):
    """Binary Focal Loss + Soft Dice Loss, summed (not weighted —
    matches notebook behaviour where forward() returns f + d directly)."""

    def __init__(self, alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def _focal(self, logits, target):
        pred = logits.squeeze(1)
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

    def _dice(self, logits, target):
        pred = torch.sigmoid(logits)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        p_flat = pred.view(pred.size(0), -1)
        t_flat = target.view(target.size(0), -1)
        inter = (p_flat * t_flat).sum(dim=1)
        denom = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        return (1 - (2 * inter + self.smooth) / (denom + self.smooth)).mean()

    def forward(self, logits, target):
        f = self._focal(logits, target)
        d = self._dice(logits, target)
        if torch.isnan(f) or torch.isnan(d):
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        return f + d
