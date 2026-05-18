"""
VisiGuard Configuration (OPTIMIZED FOR 85%+ ACCURACY)
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

MIN_IMAGES_PER_CLASS = 15
MAX_IMAGES_PER_CLASS = 100       # 🔥 FIX: Increased to 100 to ensure stable ArcFace clusters
MAX_IDENTITIES = 100             # 🔥 FIX: Scaled to 100 for higher identity-density and faster convergence

RANDOM_SEED = 42

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

# ─────────────────────────────────────────────
# Image settings
# ─────────────────────────────────────────────

IMAGE_SIZE = (224, 224)          # 🔥 CRITICAL FIX: Upgraded from 112x112 to prevent EfficientNet spatial collapse
IMAGE_SHAPE = (*IMAGE_SIZE, 3)

# ─────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────

BACKBONE = "EfficientNetB0"      
EMBEDDING_DIM = 256              # Kept at 256 for a compact, highly-dense feature vector
L2_LAMBDA = 1e-4                 

# ─────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────

BATCH_SIZE = 32                  

PHASE1_EPOCHS = 12               
PHASE1_LR = 3e-4

PHASE2_EPOCHS = 25               
PHASE2_LR = 5e-6                 
UNFREEZE_FROM = -80              # 🔥 FIX: Unfreezing deeper (-80 layers) so backbone features can morph to human faces

# ─────────────────────────────────────────────
# ArcFace Hyperparameters
# ─────────────────────────────────────────────

ARC_MARGIN = 0.3                 # Kept at 0.3 to prevent gradient explosion during early fine-tuning
ARC_SCALE = 32.0                 

# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

EARLY_STOP_PATIENCE = 10
REDUCE_LR_FACTOR = 0.5

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.55