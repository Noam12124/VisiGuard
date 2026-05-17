"""
VisiGuard Configuration
=======================
Single source of truth for all hyperparameters, paths, and settings.
Modify values here; nothing else needs to change.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.keras")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
KAGGLE_DATASET   = "atulanandjha/lfwpeople"   # kagglehub slug
MIN_IMAGES_PER_CLASS = 20                     # filter rare identities
RANDOM_SEED      = 42

# Train / Val / Test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ─────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────
IMAGE_SIZE   = (160, 160)   # (H, W) – good trade-off for EfficientNet faces
IMAGE_SHAPE  = (160, 160, 3)

# ─────────────────────────────────────────────
# Data Augmentation (applied only during training)
# ─────────────────────────────────────────────
AUG_ROTATION_RANGE   = 15      # degrees
AUG_BRIGHTNESS_DELTA = 0.25    # ± fraction
AUG_CONTRAST_LOWER   = 0.8     # lower bound of contrast factor
AUG_CONTRAST_UPPER   = 1.2     # upper bound of contrast factor
AUG_ZOOM_RANGE       = 0.10    # ± fraction of image size
AUG_HFLIP            = True    # horizontal flip

# ─────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────
BACKBONE          = "EfficientNetB0"   # pretrained ImageNet backbone
EMBEDDING_DIM     = 512               # first dense layer size
DROPOUT_1         = 0.40              # after embedding layer
DENSE_2_DIM       = 256               # second dense layer size
DROPOUT_2         = 0.30              # after second dense layer
L2_LAMBDA         = 1e-4             # L2 weight regularisation

# ─────────────────────────────────────────────
# Training – Phase 1 (frozen backbone, warm-up)
# ─────────────────────────────────────────────
PHASE1_EPOCHS     = 15
PHASE1_LR         = 1e-3
BATCH_SIZE        = 32

# ─────────────────────────────────────────────
# Training – Phase 2 (partial unfreeze, fine-tune)
# ─────────────────────────────────────────────
PHASE2_EPOCHS     = 40
PHASE2_LR         = 5e-5
UNFREEZE_FROM     = -30    # unfreeze last N layers of backbone

# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────
EARLY_STOP_PATIENCE   = 10    # epochs with no improvement before stopping
REDUCE_LR_FACTOR      = 0.4   # factor to reduce LR on plateau
REDUCE_LR_PATIENCE    = 4     # epochs before reducing LR
REDUCE_LR_MIN         = 1e-7  # minimum learning rate floor

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
CONFUSION_MATRIX_FIGSIZE = (18, 16)
CURVE_FIGSIZE            = (12, 5)

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.50   # below this → "Unknown"
