"""
VisiGuard Configuration (FINAL FIXED)
"""

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# Directory paths (REQUIRED BY utils.ensure_dirs)
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model files
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.keras")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ─────────────────────────────────────────────
# Dataset settings
# ─────────────────────────────────────────────

KAGGLE_DATASET = "hearfool/vggface2"

MIN_IMAGES_PER_CLASS = 10
MAX_IMAGES_PER_CLASS = 60
MAX_IDENTITIES = 2000

RANDOM_SEED = 42

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

# ─────────────────────────────────────────────
# Image settings
# ─────────────────────────────────────────────

IMAGE_SIZE = (160, 160)
IMAGE_SHAPE = (*IMAGE_SIZE, 3)   # ← REQUIRED FIX

# PK sampling (triplet loss training)
P_ID = 16
K_IMG = 4

# ─────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────

AUG_ROTATION_RANGE = 15
AUG_BRIGHTNESS_DELTA = 0.25
AUG_CONTRAST_LOWER = 0.8
AUG_CONTRAST_UPPER = 1.2
AUG_ZOOM_RANGE = 0.10
AUG_HFLIP = True

# ─────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────

BACKBONE = "EfficientNetB0"
EMBEDDING_DIM = 512

# ─────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────

PHASE1_EPOCHS = 15
PHASE1_LR = 3e-4
BATCH_SIZE = 32

PHASE2_EPOCHS = 40
PHASE2_LR = 1e-5
UNFREEZE_FROM = -30

# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

EARLY_STOP_PATIENCE = 10
REDUCE_LR_FACTOR = 0.5

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.60

L2_LAMBDA = 5e-4
