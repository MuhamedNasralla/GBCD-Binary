"""
dataset.py
HybridSiam-CD+ (Berlin) — Dataset Loader

Loads bi-temporal orthophoto tile pairs and binary change masks from
the GBCD-Binary dataset (before/after/label folder structure).
Supports filtering to non-empty tiles only for training (Berlin
training used 637/1,417 non-empty tiles), and Albumentations-based
augmentation matching the configuration used during training.
"""

import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config


def find_label(label_dir, stem, extensions=config.EXTENSIONS):
    for ext in extensions:
        p = os.path.join(label_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def get_train_augmentation():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.5),
            A.GaussNoise(p=0.2),
        ],
        additional_targets={"image2": "image"},
    )


def get_val_augmentation():
    return None  # resizing + normalisation only, no augmentation


class GBCDDataset(Dataset):
    """GBCD-Binary dataset loader.

    Expects a folder structure of:
        root/
          before/  *.png
          after/   *.png
          label/   *.png  (binary masks, 0=no-change, 255=change)
    """

    def __init__(
        self,
        before_dir,
        after_dir,
        label_dir,
        img_size=config.IMG_SIZE,
        augment=False,
        only_nonempty=False,
    ):
        self.before_dir = before_dir
        self.after_dir = after_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = get_train_augmentation() if augment else None

        all_names = sorted(
            f for f in os.listdir(before_dir) if f.lower().endswith(config.EXTENSIONS)
        )

        self.samples = []
        for name in all_names:
            stem = Path(name).stem
            a_path = os.path.join(after_dir, name)
            l_path = find_label(label_dir, stem)
            if not os.path.exists(a_path) or l_path is None:
                continue
            self.samples.append((name, a_path, l_path))

        if only_nonempty:
            self.samples = [
                s for s in self.samples if self._has_change(s[2])
            ]

        print(
            f"  Dataset loaded: {len(self.samples)} samples "
            f"({'non-empty only' if only_nonempty else 'all tiles'})"
        )

    @staticmethod
    def _has_change(label_path):
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        return (mask > 127).any()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, a_path, l_path = self.samples[idx]
        b_path = os.path.join(self.before_dir, name)

        before = cv2.cvtColor(cv2.imread(b_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        after = cv2.cvtColor(cv2.imread(a_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        label = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)

        before = cv2.resize(before, (self.img_size, self.img_size))
        after = cv2.resize(after, (self.img_size, self.img_size))
        label = cv2.resize(
            label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
        )
        label = (label > 127).astype(np.float32)

        if self.augment is not None:
            augmented = self.augment(image=before, image2=after, mask=label)
            before = augmented["image"]
            after = augmented["image2"]
            label = augmented["mask"]

        before = self._normalise(before)
        after = self._normalise(after)

        before_t = torch.from_numpy(before).permute(2, 0, 1).float()
        after_t = torch.from_numpy(after).permute(2, 0, 1).float()
        label_t = torch.from_numpy(label).unsqueeze(0).float()

        return before_t, after_t, label_t

    @staticmethod
    def _normalise(img):
        img = img.astype(np.float32) / 255.0
        mean = np.array(config.MEAN, dtype=np.float32)
        std = np.array(config.STD, dtype=np.float32)
        return (img - mean) / std


def build_dataloaders():
    """Builds train and validation DataLoaders using paths and settings
    from config.py."""
    from torch.utils.data import DataLoader

    train_set = GBCDDataset(
        config.TRAIN_BEFORE,
        config.TRAIN_AFTER,
        config.TRAIN_LABEL,
        img_size=config.IMG_SIZE,
        augment=True,
        only_nonempty=config.ONLY_NONEMPTY_TRAIN,
    )
    val_set = GBCDDataset(
        config.VAL_BEFORE,
        config.VAL_AFTER,
        config.VAL_LABEL,
        img_size=config.IMG_SIZE,
        augment=False,
        only_nonempty=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train : {len(train_set)} samples (non-empty only)")
    print(f"Val   : {len(val_set)} samples")

    return train_loader, val_loader
