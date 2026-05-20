"""
config.py — Central configuration for Face Recognition system.
All hyperparameters live here; import this everywhere else.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR          = "/content/face_recognition_project"   # Colab root
DATA_DIR          = os.path.join(BASE_DIR, "data", "faces")
GALLERY_DIR       = os.path.join(BASE_DIR, "data", "gallery")
CHECKPOINT_DIR    = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR           = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR        = os.path.join(BASE_DIR, "outputs")

BEST_EMBEDDING_MODEL = os.path.join(CHECKPOINT_DIR, "best_embedding_model.keras")
BEST_TRAIN_MODEL     = os.path.join(CHECKPOINT_DIR, "best_train_model.keras")
YOLO_WEIGHTS         = os.path.join(CHECKPOINT_DIR, "yolov8n_face.pt")

# ── Image ──────────────────────────────────────────────────────────────────
IMAGE_SIZE      = (112, 112)          # Standard for ArcFace / face recognition
EMBEDDING_DIM   = 512
NUM_CHANNELS    = 3

# ── Dataset ────────────────────────────────────────────────────────────────
MIN_IMAGES_PER_CLASS  = 5
VALIDATION_SPLIT      = 0.15
TEST_SPLIT            = 0.10
RANDOM_SEED           = 42

# ── Model — Backbone ───────────────────────────────────────────────────────
# Options: "efficientnetv2s", "convnext_tiny", "resnet50v2"
# EfficientNetV2-S chosen: best accuracy/compute tradeoff for 112×112 faces.
BACKBONE         = "efficientnetv2s"
DROPOUT_RATE     = 0.4
L2_REGULARIZER   = 5e-4

# ── ArcFace ────────────────────────────────────────────────────────────────
ARCFACE_MARGIN   = 0.5    # Angular margin in radians (~28.6°)
ARCFACE_SCALE    = 64.0   # Feature scale / temperature

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE           = 32
WARMUP_EPOCHS        = 15     # Phase 1: backbone frozen
FINETUNE_EPOCHS      = 35     # Phase 2: top layers unfrozen
UNFREEZE_TOP_LAYERS  = 80     # How many layers to unfreeze in phase 2

WARMUP_LR            = 1e-3
FINETUNE_LR          = 1e-4
MIN_LR               = 1e-7

GRADIENT_CLIP_NORM   = 1.0
EARLY_STOPPING_PATIENCE   = 10
REDUCE_LR_PATIENCE        = 5
REDUCE_LR_FACTOR          = 0.3

# Mixed precision: auto-detected at runtime; set False to force disable.
MIXED_PRECISION      = True

# ── Augmentation ───────────────────────────────────────────────────────────
AUGMENT_BRIGHTNESS   = 0.2
AUGMENT_CONTRAST     = 0.2
AUGMENT_SATURATION   = 0.15
AUGMENT_HUE          = 0.05
AUGMENT_FLIP         = True
AUGMENT_ROTATION     = 15      # degrees
AUGMENT_ZOOM         = 0.10

# ── Inference / Matching ───────────────────────────────────────────────────
SAME_PERSON_THRESHOLD   = 0.55   # Cosine similarity cutoff
FACE_CONF_THRESHOLD     = 0.50   # YOLOv8 minimum detection confidence
MIN_FACE_SIZE           = 20     # Minimum face dimension (pixels)

# ── Alignment ──────────────────────────────────────────────────────────────
# Eye landmark indices in YOLOv8-face keypoints (5-point model)
# Order: [left-eye, right-eye, nose, mouth-left, mouth-right]
LEFT_EYE_IDX    = 0
RIGHT_EYE_IDX   = 1
DESIRED_LEFT_EYE  = (0.35, 0.40)   # Fraction of output image dimensions
DESIRED_RIGHT_EYE = (0.65, 0.40)

# ── Evaluation ─────────────────────────────────────────────────────────────
EVAL_PAIRS_PER_CLASS = 50    # Pairs per identity for verification eval
EER_THRESHOLD_STEPS  = 1000  # Resolution of EER search
