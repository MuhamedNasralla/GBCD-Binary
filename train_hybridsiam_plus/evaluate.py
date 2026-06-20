"""
evaluate.py
HybridSiam-CD+ (Berlin) — Evaluation Script

Evaluates a trained HybridSiam-CD+ checkpoint on the GBCD-Binary test
set, reporting all 8 metrics: Accuracy, IoU, F1, Precision, Recall,
Specificity, Cohen's Kappa, and MCC.

Usage:
    python evaluate.py --weights /path/to/best_model.pth
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

import config
from dataset import GBCDDataset
from metrics import RunningConfusionMatrix
from model import HybridSiamPlus


def load_model(weights_path, device):
    model = HybridSiamPlus(pretrained_base=False)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict"):
        if isinstance(state, dict) and k in state:
            state = state[k]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    cm = RunningConfusionMatrix()
    for before, after, label in loader:
        before = before.to(device)
        after = after.to(device)
        label = label.to(device)
        logits = model(before, after)
        cm.update(logits, label, is_logits=True)
    return cm.compute()


def print_results(metrics, title="Test Set Results"):
    print()
    print("=" * 55)
    print(f"  {title}")
    print("=" * 55)
    print(f"  Accuracy     : {metrics['accuracy'] * 100:.2f}%")
    print(f"  IoU          : {metrics['iou'] * 100:.2f}%")
    print(f"  F1 Score     : {metrics['f1'] * 100:.2f}%")
    print(f"  Precision    : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall       : {metrics['recall'] * 100:.2f}%")
    print(f"  Specificity  : {metrics['specificity'] * 100:.2f}%")
    print(f"  Cohen's Kappa: {metrics['kappa'] * 100:.2f}%")
    print(f"  MCC          : {metrics['mcc'] * 100:.2f}%")
    print("-" * 55)
    print(f"  TP={metrics['tp']:,}  FP={metrics['fp']:,}  "
          f"FN={metrics['fn']:,}  TN={metrics['tn']:,}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Evaluate HybridSiam-CD+ (Berlin)")
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(config.SAVE_DIR, "best_model.pth"),
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on",
    )
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}")

    if args.split == "test":
        before_dir, after_dir, label_dir = (
            config.TEST_BEFORE,
            config.TEST_AFTER,
            config.TEST_LABEL,
        )
    else:
        before_dir, after_dir, label_dir = (
            config.VAL_BEFORE,
            config.VAL_AFTER,
            config.VAL_LABEL,
        )

    dataset = GBCDDataset(
        before_dir,
        after_dir,
        label_dir,
        img_size=config.IMG_SIZE,
        augment=False,
        only_nonempty=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = load_model(args.weights, device)
    metrics = evaluate(model, loader, device)
    print_results(metrics, title=f"GBCD-Binary {args.split.upper()} Set Results")


if __name__ == "__main__":
    main()
