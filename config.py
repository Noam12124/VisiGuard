"""
VisiGuard Configuration (VGGFace2 Optimized v2)
==============================================
Stability + high-accuracy tuning for VGGFace2.
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
# Dataset (VGGFace2 SAFE CONFIG)
# ─────────────────────────────────────────────
KAGGLE_DATASET   = "hearfool/vggface2"

MIN_IMAGES_PER_CLASS = 10     # 🔥 better balance (8 → 10)
MAX_IMAGES_PER_CLASS = 60     # 🔥 IMPORTANT (YOU MUST USE THIS IN dataset.py)

MAX_IDENTITIES = 2000         # safety cap (must be enforced in scan)

RANDOM_SEED = 42

TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10

# ─────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────
IMAGE_SIZE  = (160, 160)
IMAGE_SHAPE = (160, 160, 3)

# ─────────────────────────────────────────────
# Augmentation (FACE-OPTIMIZED)
# ─────────────────────────────────────────────
AUG_ROTATION_RANGE   = 15     # reduced → faces are sensitive
AUG_BRIGHTNESS_DELTA = 0.25
AUG_CONTRAST_LOWER   = 0.8
AUG_CONTRAST_UPPER   = 1.2
AUG_ZOOM_RANGE       = 0.10   # reduced → prevents distortion
AUG_HFLIP            = True

# ─────────────────────────────────────────────
# Model architecture (UNCHANGED)
# ─────────────────────────────────────────────
BACKBONE       = "EfficientNetB0"
EMBEDDING_DIM  = 512
DROPOUT_1      = 0.40
DENSE_2_DIM    = 256
DROPOUT_2      = 0.30
L2_LAMBDA      = 1e-4

# ─────────────────────────────────────────────
# Training – Phase 1 (IMPORTANT FIX)
# ─────────────────────────────────────────────
PHASE1_EPOCHS  = 15
PHASE1_LR      = 3e-4     # 🔥 safer than 5e-4
BATCH_SIZE     = 32

# ─────────────────────────────────────────────
# Training – Phase 2 (fine tuning)
# ─────────────────────────────────────────────
PHASE2_EPOCHS  = 40
PHASE2_LR      = 1e-5     # 🔥 more stable for face embeddings
UNFREEZE_FROM  = -30      # 🔥 safer generalization (was -40)

# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────
EARLY_STOP_PATIENCE = 10
REDUCE_LR_FACTOR    = 0.5
REDUCE_LR_PATIENCE  = 3
REDUCE_LR_MIN       = 1e-7

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
CONFUSION_MATRIX_FIGSIZE = (18, 16)
CURVE_FIGSIZE            = (12, 5)

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # 🔥 slightly stricter = fewer false positives