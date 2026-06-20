"""
dataset.py
HybridSiam-CD+ (Combined) — Multi-Dataset Loader

Loads and combines four datasets with differing folder conventions:
  - GBCD-Binary   : before / after / label
  - FOTBCD-Binary : before / after / label
  - LEVIR-CD+     : A / B / label   (no official val split — 10% auto-split)
  - WHU-CD        : A / B / OUT

Matches the exact ChangeDetectionDataset and combination logic used in
the combined multi-domain training notebook (1,769+25,908+574+6,096
≈ 33,994 combined training tiles after the LEVIR-CD+ auto-split).
"""

import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import config


def find_label(label_dir, stem, extensions=config.EXTENSIONS):
    for ext in extensions:
        p = os.path.join(label_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


class ChangeDetectionDataset(Dataset):
    """Generic bi-temporal change detection dataset, configurable to any
    before/after/label subfolder naming convention."""

    def __init__(
        self,
        base_path,
        split_name,
        before_sub,
        after_sub,
        label_sub,
        img_size=config.IMG_SIZE,
        augment=True,
        sample_indices=None,
        dataset_name="",
    ):
        super().__init__()
        self.img_size = img_size
        self.dataset_name = dataset_name
        split_dir = os.path.join(base_path, split_name)
        before_dir = os.path.join(split_dir, before_sub)
        after_dir = os.path.join(split_dir, after_sub)
        label_dir = os.path.join(split_dir, label_sub)

        for d in (before_dir, after_dir, label_dir):
            if not os.path.exists(d):
                raise FileNotFoundError(f"Not found: {d}")

        all_samples = []
        for name in sorted(os.listdir(before_dir)):
            if not name.lower().endswith(config.EXTENSIONS):
                continue
            stem = Path(name).stem
            a_p = os.path.join(after_dir, name)
            l_p = find_label(label_dir, stem)
            if os.path.exists(a_p) and l_p:
                all_samples.append((os.path.join(before_dir, name), a_p, l_p))

        if sample_indices is not None:
            self.samples = [all_samples[i] for i in sample_indices]
        else:
            self.samples = all_samples

        if not self.samples:
            raise RuntimeError(f"No samples in {split_dir}")

        if augment:
            self.tf = A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.3),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
                    A.GaussNoise(p=0.2),
                    A.Normalize(mean=config.MEAN, std=config.STD),
                    ToTensorV2(),
                ],
                additional_targets={"image2": "image"},
                is_check_shapes=False,
            )
        else:
            self.tf = A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=config.MEAN, std=config.STD),
                    ToTensorV2(),
                ],
                additional_targets={"image2": "image"},
                is_check_shapes=False,
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b_p, a_p, l_p = self.samples[idx]
        before = cv2.cvtColor(cv2.imread(b_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        after = cv2.cvtColor(cv2.imread(a_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        label = cv2.imread(l_p, cv2.IMREAD_GRAYSCALE)

        before = cv2.resize(before, (self.img_size, self.img_size))
        after = cv2.resize(after, (self.img_size, self.img_size))
        label = cv2.resize(
            label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
        )
        mask = (label > 127).astype(np.uint8)

        aug = self.tf(image=before, image2=after, mask=mask)
        return {
            "A": aug["image"].float(),
            "B": aug["image2"].float(),
            "mask": aug["mask"].float(),
        }


def build_combined_datasets(seed=config.SEED, verbose=True):
    """Builds and combines train/val datasets from all 4 configured
    datasets. LEVIR-CD+ has no official val split, so 10% of its
    training tiles are randomly held out for validation."""
    random.seed(seed)
    np.random.seed(seed)

    train_datasets = []
    val_datasets = []

    for cfg in config.DATASET_CONFIGS:
        name = cfg["name"]
        if verbose:
            print(f"\nBuilding {name} ...")

        if cfg["val_split"] is None:
            full_ds = ChangeDetectionDataset(
                cfg["base"], cfg["train_split"],
                cfg["before"], cfg["after"], cfg["label"],
                augment=False, dataset_name=name,
            )
            n = len(full_ds.samples)
            indices = list(range(n))
            random.shuffle(indices)
            n_val = max(1, int(n * config.VAL_SPLIT_RATIO))
            val_idx = indices[:n_val]
            trn_idx = indices[n_val:]

            train_ds = ChangeDetectionDataset(
                cfg["base"], cfg["train_split"],
                cfg["before"], cfg["after"], cfg["label"],
                augment=True, sample_indices=trn_idx, dataset_name=name,
            )
            val_ds = ChangeDetectionDataset(
                cfg["base"], cfg["train_split"],
                cfg["before"], cfg["after"], cfg["label"],
                augment=False, sample_indices=val_idx, dataset_name=name,
            )
            if verbose:
                print(f"  Auto-split: train={len(train_ds)} val={len(val_ds)}")
        else:
            train_ds = ChangeDetectionDataset(
                cfg["base"], cfg["train_split"],
                cfg["before"], cfg["after"], cfg["label"],
                augment=True, dataset_name=name,
            )
            val_ds = ChangeDetectionDataset(
                cfg["base"], cfg["val_split"],
                cfg["before"], cfg["after"], cfg["label"],
                augment=False, dataset_name=name,
            )
            if verbose:
                print(f"  train={len(train_ds)} val={len(val_ds)}")

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"Combined train : {len(combined_train):,} tiles")
        print(f"Combined val   : {len(combined_val):,} tiles")
        print("\nBreakdown:")
        for cfg, td, vd in zip(config.DATASET_CONFIGS, train_datasets, val_datasets):
            print(f"  {cfg['name']:<20} train={len(td):>5}  val={len(vd):>5}")

    return combined_train, combined_val


def build_dataloaders():
    """Builds train and validation DataLoaders for combined training."""
    combined_train, combined_val = build_combined_datasets()

    train_loader = DataLoader(
        combined_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val   loader: {len(val_loader)} batches")
    print(f"Batch size  : {config.BATCH_SIZE}")
    print(f"Each epoch  : ~{len(train_loader) * config.BATCH_SIZE:,} images")

    return train_loader, val_loader
