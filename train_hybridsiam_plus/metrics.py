"""
metrics.py
HybridSiam-CD+ — Evaluation Metrics

All metrics are computed at the pixel level from the confusion matrix
counts (TP, FP, FN, TN), aggregated either per-batch (training/
validation loop) or across an entire test set (evaluate.py).

Definitions follow standard remote sensing change detection evaluation
conventions (Congalton, 1991; Chen and Shi, 2020; Chicco and Jurman,
2020).
"""

import numpy as np
import torch


def confusion_counts(preds, targets):
    """preds, targets: binary tensors/arrays of identical shape.
    Returns TP, FP, FN, TN as floats."""
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    preds = preds.astype(bool)
    targets = targets.astype(bool)

    tp = float(np.logical_and(preds, targets).sum())
    fp = float(np.logical_and(preds, ~targets).sum())
    fn = float(np.logical_and(~preds, targets).sum())
    tn = float(np.logical_and(~preds, ~targets).sum())
    return tp, fp, fn, tn


def compute_metrics(tp, fp, fn, tn, eps=1e-8):
    """Computes all 8 evaluation metrics from confusion matrix counts."""
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


def metrics_from_predictions(preds, targets, threshold=0.5, is_logits=True):
    """Convenience wrapper: takes raw model output (logits or probs) and
    ground-truth labels, applies sigmoid + threshold if needed, and
    returns the full metrics dict."""
    if is_logits:
        probs = torch.sigmoid(preds) if isinstance(preds, torch.Tensor) else preds
    else:
        probs = preds

    if isinstance(probs, torch.Tensor):
        binary_preds = (probs > threshold).float()
    else:
        binary_preds = (probs > threshold).astype(np.float32)

    tp, fp, fn, tn = confusion_counts(binary_preds, targets)
    return compute_metrics(tp, fp, fn, tn)


class RunningConfusionMatrix:
    """Accumulates TP/FP/FN/TN across multiple batches (e.g. across an
    entire validation epoch or test set) so that metrics can be computed
    once at the end rather than averaged per-batch."""

    def __init__(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    def update(self, preds, targets, threshold=0.5, is_logits=True):
        if is_logits:
            probs = torch.sigmoid(preds) if isinstance(preds, torch.Tensor) else preds
        else:
            probs = preds

        if isinstance(probs, torch.Tensor):
            binary_preds = (probs > threshold).float()
        else:
            binary_preds = (probs > threshold).astype(np.float32)

        tp, fp, fn, tn = confusion_counts(binary_preds, targets)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

    def compute(self):
        return compute_metrics(self.tp, self.fp, self.fn, self.tn)

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0
