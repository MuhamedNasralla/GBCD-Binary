"""
config.py
HybridSiam-CD+ (Combined) — Configuration

Joint multi-domain training across four datasets: GBCD-Binary (Berlin),
FOTBCD-Binary (France), LEVIR-CD+ (China), and WHU-CD (Wuhan, China).
Matches the exact configuration used to train the combined model that
achieved best val IoU = 0.8015 at epoch 30 (early stopped at epoch 40).
"""

import os

# ───────────────────────── Paths ─────────────────────────
# NOTE: update DATA_ROOT_DIR and PRETRAINED_WEIGHTS to your local environment.
# Original Colab paths were under /content/ after unzipping each dataset
# from /content/drive/MyDrive/GBCD/<name>.zip
DATA_ROOT_DIR       = "/content"
PRETRAINED_WEIGHTS  = os.path.join(DATA_ROOT_DIR, "model.pth")
SAVE_DIR            = os.path.join(DATA_ROOT_DIR, "runs", "combined")

DRIVE_BEST_MODEL_PATH = "/content/drive/MyDrive/GBCD/best_model_combined.pth"
DRIVE_LOG_PATH         = "/content/drive/MyDrive/GBCD/combined_training_log.csv"

# ───────────────────────── Dataset composition ─────────────────────────
# Each dataset uses a different folder naming convention for the
# before/after/label triplet — handled transparently by dataset.py.
DATASET_CONFIGS = [
    {
        "name": "GBCD-Binary",
        "base": os.path.join(DATA_ROOT_DIR, "GBCD-Binary"),
        "before": "before",
        "after": "after",
        "label": "label",
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
    },
    {
        "name": "FOTBCD-Binary",
        "base": os.path.join(DATA_ROOT_DIR, "FOTBCD-Binary"),
        "before": "before",
        "after": "after",
        "label": "label",
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
    },
    {
        "name": "LEVIR-CD+",
        "base": os.path.join(DATA_ROOT_DIR, "LEVIR-CD+"),
        "before": "A",
        "after": "B",
        "label": "label",
        "train_split": "train",
        "val_split": None,  # no official val split — 10% auto-split from train
        "test_split": "test",
    },
    {
        "name": "whu-cd",
        "base": os.path.join(DATA_ROOT_DIR, "whu-cd"),
        "before": "A",
        "after": "B",
        "label": "OUT",
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
    },
]

# ───────────────────────── Data ─────────────────────────
IMG_SIZE   = 512
EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
VAL_SPLIT_RATIO = 0.10  # used only for LEVIR-CD+ (no official val split)

# ───────────────────────── Training ─────────────────────────
BATCH_SIZE  = 64    # A100 80GB; reduce (e.g. 8) for smaller GPUs
NUM_WORKERS = 8
NUM_EPOCHS  = 40
PATIENCE    = 8
WARMUP_EPOCHS = 3

# ───────────────────────── Optimiser / LR groups ─────────────────────────
LR_DECODER = 1e-5    # decoder + ResNet34 spatial branch (pretrained, fine-tune gently)
LR_CBAM    = 1e-4    # CBAM attention modules (randomly initialised, new)
LR_FUSION  = 1e-4    # learnable temporal fusion layer (randomly initialised, new)
WEIGHT_DECAY = 1e-4
GRAD_CLIP  = 1.0

# ───────────────────────── Architecture ─────────────────────────
FREEZE_VIT = True    # freeze the DINOv3 ViT-L semantic encoder

# ───────────────────────── Loss (Focal + Dice) ─────────────────────────
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.5

# ───────────────────────── Mixed precision ─────────────────────────
USE_AMP = True  # Automatic Mixed Precision — safe on 80GB GPU

# ───────────────────────── Misc ─────────────────────────
SEED = 42
DEVICE = "cuda"  # falls back to cpu automatically in train.py if unavailable
