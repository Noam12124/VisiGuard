"""
VisiGuard Configuration (FAST + STABLE)
"""

import os

# ─────────────────────────────────────────────
# Base directories
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.keras")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ─────────────────────────────────────────────
# Dataset settings
# ─────────────────────────────────────────────

KAGGLE_DATASET = "hearfool/vggface2"

MIN_IMAGES_PER_CLASS = 10
MAX_IMAGES_PER_CLASS = 30        # ↓ Faster
MAX_IDENTITIES = 300             # ↓ Much faster

RANDOM_SEED = 42

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

# ─────────────────────────────────────────────
# Image settings
# ─────────────────────────────────────────────

IMAGE_SIZE = (112, 112)          # ↓ ArcFace standard + faster
IMAGE_SHAPE = (*IMAGE_SIZE, 3)

# ─────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────

BACKBONE = "EfficientNetB0"      # Can switch to MobileNetV3 for even more speed
EMBEDDING_DIM = 512
L2_LAMBDA = 5e-4

# ─────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────

BATCH_SIZE = 64                  # ↑ Faster on T4
PHASE1_EPOCHS = 10               # Enough for frozen backbone
PHASE1_LR = 3e-4

PHASE2_EPOCHS = 20               # Fine-tuning
PHASE2_LR = 1e-5
UNFREEZE_FROM = -30

# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

EARLY_STOP_PATIENCE = 8
REDUCE_LR_FACTOR = 0.5

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.60
