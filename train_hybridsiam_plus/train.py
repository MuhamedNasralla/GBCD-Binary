"""
train.py
HybridSiam-CD+ (Berlin) — Training Script

Fine-tunes HybridSiam-CD+ (CBAM + learnable temporal fusion + Focal-Dice
loss) on the GBCD-Binary (Berlin) dataset, starting from the pretrained
HybridSiam-CD (FOTBCD) weights with a frozen ViT backbone.

Usage:
    python train.py
"""

import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import config
from dataset import build_dataloaders
from losses import FocalDiceLoss
from metrics import RunningConfusionMatrix
from model import HybridSiamPlus, load_pretrained_weights


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, optimizer, scheduler, device, train=True):
    model.train() if train else model.eval()
    running_loss = 0.0
    cm = RunningConfusionMatrix()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for before, after, label in loader:
            before = before.to(device, non_blocking=True)
            after = after.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            logits = model(before, after)
            loss, _ = criterion(logits, label)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRAD_CLIP_NORM
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item() * before.size(0)
            cm.update(logits, label, is_logits=True)

    avg_loss = running_loss / len(loader.dataset)
    metrics = cm.compute()
    return avg_loss, metrics


def main():
    set_seed(config.SEED)
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(config.SAVE_DIR, exist_ok=True)

    print("=" * 65)
    print("  HybridSiam-CD+  |  CBAM + Temporal Fusion + Focal Loss")
    print("=" * 65)
    print(f"Device        : {device}")
    print(f"Data root     : {config.DATA_ROOT}")
    print(f"Pretrained    : {config.PRETRAINED_WEIGHTS}")
    print(f"Save dir      : {config.SAVE_DIR}")
    print(f"IMG_SIZE      : {config.IMG_SIZE}   BATCH_SIZE : {config.BATCH_SIZE}")
    print(f"Epochs        : {config.EPOCHS}  patience={config.PATIENCE}")
    print(f"Freeze ViT    : {config.FREEZE_VIT}")
    print(f"Warmup epochs : {config.WARMUP_EPOCHS}")
    print("-" * 67)

    # ── Data ──
    train_loader, val_loader = build_dataloaders()
    print()

    # ── Model ──
    print("Building HybridSiam-CD+ ...")
    model = HybridSiamPlus(pretrained_base=False)
    model = load_pretrained_weights(model, config.PRETRAINED_WEIGHTS, device="cpu")

    if config.FREEZE_VIT:
        model.freeze_vit_backbone()
        print("[VRAM] ViT backbone frozen.")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(
        f"Parameters : {total_params:,} total | "
        f"{trainable_params:,} trainable | {frozen_params:,} frozen"
    )

    # ── Optimiser ──
    param_groups = model.get_param_groups(
        lr_decoder=config.LR_DECODER,
        lr_cbam=config.LR_CBAM,
        lr_fusion=config.LR_FUSION,
    )
    optimizer = AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)

    print("[Optimizer]")
    for g in param_groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:<22} : {n_params:,}  LR={g['lr']}")
    print()

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    # ── Loss ──
    criterion = FocalDiceLoss(
        focal_alpha=config.FOCAL_ALPHA,
        focal_gamma=config.FOCAL_GAMMA,
        focal_weight=config.FOCAL_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
    )

    print("=" * 65)
    print(f"Training HybridSiam-CD+ on {config.IMG_SIZE}x{config.IMG_SIZE} Berlin data")
    print("=" * 65)
    print()

    # ── Training loop ──
    best_iou = 0.0
    no_improve = 0
    log_path = os.path.join(config.SAVE_DIR, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch", "train_loss", "train_iou", "train_f1",
                "val_loss", "val_iou", "val_f1", "val_precision", "val_recall",
                "lr", "time_sec",
            ]
        )

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()

        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, train=True
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, scheduler, device, train=False
        )

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{config.EPOCHS}  "
            f"Train loss={train_loss:.4f} IoU={train_metrics['iou']:.4f} "
            f"F1={train_metrics['f1']:.4f}  |  "
            f"Val loss={val_loss:.4f} IoU={val_metrics['iou']:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f}  "
            f"LR={current_lr:.2e}  [{elapsed:.0f}s]"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch, train_loss, train_metrics["iou"], train_metrics["f1"],
                    val_loss, val_metrics["iou"], val_metrics["f1"],
                    val_metrics["precision"], val_metrics["recall"],
                    current_lr, round(elapsed),
                ]
            )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            no_improve = 0
            torch.save(
                model.state_dict(),
                os.path.join(config.SAVE_DIR, "best_model.pth"),
            )
            print(f"  \u2713 New best IoU: {best_iou:.4f} \u2192 saved best_model.pth")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{config.PATIENCE})")

        if no_improve >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    torch.save(
        model.state_dict(),
        os.path.join(config.SAVE_DIR, "final_model.pth"),
    )

    print()
    print("=" * 65)
    print(f"Training complete.  Best val IoU : {best_iou:.4f}")
    print(f"Best model   \u2192 {os.path.join(config.SAVE_DIR, 'best_model.pth')}")
    print(f"Final model  \u2192 {os.path.join(config.SAVE_DIR, 'final_model.pth')}")
    print(f"Log          \u2192 {log_path}")
    print("=" * 65)
    print()
    print("Next: run evaluate.py pointing to:")
    print(f"  {os.path.join(config.SAVE_DIR, 'best_model.pth')}")


if __name__ == "__main__":
    main()
