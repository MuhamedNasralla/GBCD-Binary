"""
train.py
HybridSiam-CD+ (Combined) — Training Script

Joint multi-domain training across GBCD-Binary, FOTBCD-Binary, LEVIR-CD+,
and WHU-CD (33,994 combined training tiles). Uses Automatic Mixed
Precision, gradient clipping, NaN-batch skipping, and per-epoch
checkpoint resume to survive Colab session interruptions.

Matches the exact training loop from the combined training notebook,
which reached best val IoU = 0.8015 at epoch 30 before early stopping
at epoch 40.

Usage:
    python train.py
"""

import os
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import build_dataloaders
from losses import FocalDiceLoss
from metrics import Metrics
from model import HybridSiamPlus


def set_seed(seed=config.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    metrics = Metrics()
    total_loss = 0.0
    counted = 0
    for batch in loader:
        before = batch["A"].to(device)
        after = batch["B"].to(device)
        mask = batch["mask"].to(device)
        if mask.sum() == 0:
            continue
        with torch.autocast(device_type="cuda", enabled=config.USE_AMP):
            pred = model(before, after)
            loss = criterion(pred, mask)
        if not torch.isnan(loss):
            total_loss += loss.item()
            counted += 1
        metrics.update(pred, mask)
    m = metrics.compute()
    m["loss"] = total_loss / max(counted, 1)
    model.train()
    return m


def main():
    set_seed()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # ── Data ──
    train_loader, val_loader = build_dataloaders()

    # ── Model ──
    print("\nBuilding HybridSiam-CD+ ...")
    model = HybridSiamPlus(
        pretrained_path=config.PRETRAINED_WEIGHTS, freeze_vit=config.FREEZE_VIT
    )
    model = model.to(device)

    # ── Optimiser ──
    param_groups = model.get_param_groups(
        lr_decoder=config.LR_DECODER, lr_cbam=config.LR_CBAM, lr_fusion=config.LR_FUSION
    )
    optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Parameters : {n_total:,} total | {n_trainable:,} trainable | "
        f"{n_total - n_trainable:,} frozen"
    )
    for g in param_groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"{g['name'].capitalize():<11}: {n_params:,}  LR={g['lr']}")

    # ── Scheduler ──
    def lr_lambda(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return (epoch + 1) / config.WARMUP_EPOCHS
        progress = (epoch - config.WARMUP_EPOCHS) / max(
            config.NUM_EPOCHS - config.WARMUP_EPOCHS, 1
        )
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = FocalDiceLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)

    # ── Resume from checkpoint if present ──
    log_path = os.path.join(config.SAVE_DIR, "training_log.csv")
    resume_path = os.path.join(config.SAVE_DIR, "latest_checkpoint.pth")

    start_epoch = 0
    best_iou = 0.0
    no_improve = 0

    if os.path.exists(resume_path):
        print("Resuming from checkpoint ...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt["best_iou"]
        no_improve = ckpt["no_improve"]
        print(f"  Resumed at epoch {start_epoch}, best_iou={best_iou:.4f}")
    else:
        with open(log_path, "w") as f:
            f.write(
                "epoch,train_loss,train_iou,train_f1,"
                "val_loss,val_iou,val_f1,"
                "val_precision,val_recall,lr\n"
            )

    print(f"\n{'=' * 65}")
    print("  HybridSiam-CD+ \u2014 Combined Training")
    print("  Datasets: GBCD + FOTBCD + LEVIR + WHU")
    print(f"  Train tiles : {len(train_loader.dataset):,}")
    print(f"  Val tiles   : {len(val_loader.dataset):,}")
    print(f"  Epochs      : {config.NUM_EPOCHS}  patience={config.PATIENCE}")
    print(f"  Batch size  : {config.BATCH_SIZE}  AMP: {'ON' if config.USE_AMP else 'OFF'}")
    print(f"{'=' * 65}\n")

    # ── Training loop ──
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        train_m = Metrics()
        train_loss = 0.0
        nan_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:03d}/{config.NUM_EPOCHS}", leave=False)

        for batch in pbar:
            before = batch["A"].to(device, non_blocking=True)
            after = batch["B"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=config.USE_AMP):
                pred = model(before, after)
                loss = criterion(pred, mask)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_m.update(pred.detach(), mask)
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        tm = train_m.compute()
        t_loss = train_loss / max(len(train_loader) - nan_batches, 1)
        vm = validate(model, val_loader, criterion, device)
        lr_now = scheduler.get_last_lr()[0]
        secs = time.time() - t0

        nan_msg = f" [{nan_batches} nan]" if nan_batches else ""
        print(
            f"Epoch {epoch + 1:03d}/{config.NUM_EPOCHS}  "
            f"Train loss={t_loss:.4f} IoU={tm['iou']:.4f} F1={tm['f1']:.4f}  |  "
            f"Val loss={vm['loss']:.4f} IoU={vm['iou']:.4f} "
            f"F1={vm['f1']:.4f} P={vm['precision']:.4f} R={vm['recall']:.4f}  "
            f"LR={lr_now:.2e}  [{secs:.0f}s]{nan_msg}"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch + 1},{t_loss:.6f},{tm['iou']:.6f},"
                f"{tm['f1']:.6f},{vm['loss']:.6f},{vm['iou']:.6f},"
                f"{vm['f1']:.6f},{vm['precision']:.6f},"
                f"{vm['recall']:.6f},{lr_now:.2e}\n"
            )

        if vm["iou"] > best_iou:
            best_iou = vm["iou"]
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(config.SAVE_DIR, "best_model.pth"))
            print(f"  \u2713 New best IoU: {best_iou:.4f} \u2192 saved best_model.pth")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{config.PATIENCE})")

        # Per-epoch checkpoint — survives Colab session interruptions
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_iou": best_iou,
                "no_improve": no_improve,
            },
            resume_path,
        )

        if no_improve >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}.")
            break

    print(f"\n{'=' * 65}")
    print(f"Training complete.  Best val IoU : {best_iou:.4f}")
    print(f"Best model  \u2192 {os.path.join(config.SAVE_DIR, 'best_model.pth')}")
    print(f"{'=' * 65}")

    # ── Save to Drive if available (Colab convenience) ──
    if os.path.isdir(os.path.dirname(config.DRIVE_BEST_MODEL_PATH)):
        import shutil

        shutil.copy(
            os.path.join(config.SAVE_DIR, "best_model.pth"), config.DRIVE_BEST_MODEL_PATH
        )
        shutil.copy(log_path, config.DRIVE_LOG_PATH)
        print(f"Best model saved to Drive \u2192 {config.DRIVE_BEST_MODEL_PATH}")
        print("Training log saved to Drive.")


if __name__ == "__main__":
    main()
