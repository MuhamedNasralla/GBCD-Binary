"""
metrics.py
HybridSiam-CD+ (Combined) — Evaluation Metrics

Matches the exact running-confusion-matrix Metrics class used during
combined multi-domain training, plus an extended compute_full() helper
that adds Specificity, Cohen's Kappa, and MCC for full 8-metric
reporting in evaluate.py (Congalton, 1991; Chicco and Jurman, 2020).
"""

import numpy as np
import torch


class Metrics:
    """Accumulates TP/FP/FN/TN across batches (matches notebook's
    training-loop Metrics class exactly)."""

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0.0

    @torch.no_grad()
    def update(self, logits, target):
        pred = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
        target = target.float()
        self.tp += (pred * target).sum().item()
        self.fp += (pred * (1 - target)).sum().item()
        self.fn += ((1 - pred) * target).sum().item()
        self.tn += ((1 - pred) * (1 - target)).sum().item()

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        return dict(precision=precision, recall=recall, f1=f1, iou=iou, accuracy=accuracy)

    def compute_full(self, eps=1e-8):
        """Extended metrics for evaluate.py — adds Specificity, Cohen's
        Kappa, and MCC for the full 8-metric evaluation table."""
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
        specificity = tn / (tn + fp + eps)

        total = tp + fp + fn + tn
        po = (tp + tn) / total
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total ** 2)
        kappa = (po - pe) / (1 - pe + eps)

        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / (denom + eps) if denom > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "specificity": specificity,
            "kappa": kappa,
            "mcc": mcc,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0
