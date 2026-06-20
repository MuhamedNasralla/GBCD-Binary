"""
evaluate.py
HybridSiam-CD+ (Combined) — Evaluation Script

Evaluates a trained combined-model checkpoint on the test set of any
configured dataset (GBCD-Binary, FOTBCD-Binary, LEVIR-CD+, or WHU-CD),
reporting all 8 metrics: Accuracy, IoU, F1, Precision, Recall,
Specificity, Cohen's Kappa, and MCC.

Usage:
    python evaluate.py --weights /path/to/best_model.pth --dataset GBCD-Binary
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

import config
from dataset import ChangeDetectionDataset
from metrics import Metrics
from model import HybridSiamPlus


def load_model_for_eval(weights_path, device):
    """Builds the architecture without re-loading the original
    HybridSiam-CD pretrained weights, then loads the fine-tuned
    checkpoint directly."""
    model = HybridSiamPlus(pretrained_path=config.PRETRAINED_WEIGHTS, freeze_vit=False)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict"):
        if isinstance(state, dict) and k in state:
            state = state[k]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    metrics = Metrics()
    for batch in loader:
        before = batch["A"].to(device)
        after = batch["B"].to(device)
        mask = batch["mask"].to(device)
        with torch.autocast(device_type="cuda", enabled=config.USE_AMP):
            pred = model(before, after)
        metrics.update(pred, mask)
    return metrics.compute_full()


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
    print(
        f"  TP={metrics['tp']:,}  FP={metrics['fp']:,}  "
        f"FN={metrics['fn']:,}  TN={metrics['tn']:,}"
    )
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Evaluate HybridSiam-CD+ (Combined)")
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(config.SAVE_DIR, "best_model.pth"),
        help="Path to the trained combined-model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="GBCD-Binary",
        choices=[cfg["name"] for cfg in config.DATASET_CONFIGS],
        help="Which dataset's test set to evaluate on",
    )
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}")
    print(f"Dataset : {args.dataset}")

    cfg = next(c for c in config.DATASET_CONFIGS if c["name"] == args.dataset)
    test_split = cfg.get("test_split", "test")

    dataset = ChangeDetectionDataset(
        cfg["base"], test_split,
        cfg["before"], cfg["after"], cfg["label"],
        img_size=config.IMG_SIZE,
        augment=False,
        dataset_name=cfg["name"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = load_model_for_eval(args.weights, device)
    metrics = evaluate(model, loader, device)
    print_results(metrics, title=f"{args.dataset} TEST Set Results (Combined Model)")


if __name__ == "__main__":
    main()
