"""
config.py
HybridSiam-CD+ (Berlin) — Configuration

All hyperparameters used to fine-tune HybridSiam-CD+ on the GBCD-Binary
(Berlin) dataset, starting from the pretrained HybridSiam-CD (FOTBCD) weights.
"""

import os

# ───────────────────────── Paths ─────────────────────────
# NOTE: update these to your local environment.
# Original Colab paths were:
#   PRETRAINED_WEIGHTS = '/content/model.pth'
#   DATA_ROOT           = '/content/GBCD-Binary'
#   SAVE_DIR             = '/content/runs/hybridsiam_plus'
PRETRAINED_WEIGHTS = r"C:\Users\Mohammed\My research\GBCD\model.pth"
DATA_ROOT           = r"C:\Users\Mohammed\My research\GBCD\GBCD-Binary"
SAVE_DIR             = r"C:\Users\Mohammed\My research\GBCD\runs\hybridsiam_plus"

TRAIN_BEFORE = os.path.join(DATA_ROOT, "train", "before")
TRAIN_AFTER  = os.path.join(DATA_ROOT, "train", "after")
TRAIN_LABEL  = os.path.join(DATA_ROOT, "train", "label")

VAL_BEFORE = os.path.join(DATA_ROOT, "val", "before")
VAL_AFTER  = os.path.join(DATA_ROOT, "val", "after")
VAL_LABEL  = os.path.join(DATA_ROOT, "val", "label")

TEST_BEFORE = os.path.join(DATA_ROOT, "test", "before")
TEST_AFTER  = os.path.join(DATA_ROOT, "test", "after")
TEST_LABEL  = os.path.join(DATA_ROOT, "test", "label")

# ───────────────────────── Data ─────────────────────────
IMG_SIZE          = 512
EXTENSIONS        = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
ONLY_NONEMPTY_TRAIN = True   # train only on tiles containing at least one change pixel
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ───────────────────────── Training ─────────────────────────
BATCH_SIZE    = 8      # 8 on T4 (15GB VRAM); 2 on 4GB local GPU
NUM_WORKERS   = 2
EPOCHS        = 50
PATIENCE      = 10
WARMUP_EPOCHS = 5

# ───────────────────────── Optimiser / LR groups ─────────────────────────
LR_DECODER = 1e-5    # decoder + ResNet34 spatial branch (pretrained, fine-tune gently)
LR_CBAM    = 1e-4    # CBAM attention modules (randomly initialised, new)
LR_FUSION  = 1e-4    # learnable temporal fusion layer (randomly initialised, new)
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# ───────────────────────── Architecture ─────────────────────────
FREEZE_VIT = True    # freeze the DINOv3 ViT-L semantic encoder

# ───────────────────────── Loss (Focal + Dice) ─────────────────────────
FOCAL_ALPHA  = 0.75
FOCAL_GAMMA  = 2.5
DICE_WEIGHT  = 1.0
FOCAL_WEIGHT = 1.0

# ───────────────────────── Misc ─────────────────────────
SEED = 42
DEVICE = "cuda"  # falls back to cpu automatically in train.py if unavailable
