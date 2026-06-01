"""
config.py — Central configuration for Face Recognition system.
All hyperparameters live here; import this everywhere else.

Key changes for from-scratch training (no pretrained weights):
  • MIN_IMAGES_PER_CLASS lowered to 5 → keeps all ~1,680 LFW identities.
    More identities = more diverse training signal for ArcFace.
  • Single training phase (no frozen backbone warm-up).
  • TOTAL_EPOCHS = 120.  From-scratch networks need ~3-5× more epochs
    than fine-tuned ones to converge; 120 is the sweet spot for LFW size.
  • INITIAL_LR raised to 1e-2 with linear warm-up.  Starting from
    random weights, a higher LR with warm-up outperforms a low LR.
  • ARCFACE_SCALE lowered to 32.  Smaller datasets need a gentler
    temperature; 64 (the paper default for millions of images) causes
    overconfidence on ~8K training images.
  • L2_REGULARIZER tightened to 1e-4.  Fewer parameters to regularise.
  • DROPOUT_RATE lowered to 0.3.  Less aggressive than fine-tuning since
    the model is not being pushed away from pretrained features.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = "/content/VisiGuard"
DATA_DIR          = os.path.join(BASE_DIR, "data", "faces")
GALLERY_DIR       = os.path.join(BASE_DIR, "data", "gallery")
CHECKPOINT_DIR    = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR           = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR        = os.path.join(BASE_DIR, "outputs")

BEST_EMBEDDING_MODEL = os.path.join(CHECKPOINT_DIR, "best_embedding_model.keras")
BEST_TRAIN_MODEL     = os.path.join(CHECKPOINT_DIR, "best_train_model.keras")
YOLO_WEIGHTS         = os.path.join(CHECKPOINT_DIR, "yolov8n_face.pt")

# ── Image ──────────────────────────────────────────────────────────────────
IMAGE_SIZE      = (112, 112)     # Standard for ArcFace / face recognition
EMBEDDING_DIM   = 512
NUM_CHANNELS    = 3

# ── Dataset ────────────────────────────────────────────────────────────────
# Set to 5 so we keep all ~1,680 LFW identities.  More identities means
# more decision boundaries for ArcFace to learn, which is more important
# than having many images per class when training from scratch.
MIN_IMAGES_PER_CLASS  = 5
VALIDATION_SPLIT      = 0.10   # 10% of identities held out for validation
TEST_SPLIT            = 0.10   # 10% for test
RANDOM_SEED           = 42

# ── Model — Architecture ──────────────────────────────────────────────────
# FaceResNet trained from scratch. No ImageNet backbone.
BACKBONE         = "from_scratch"   # informational tag only
DROPOUT_RATE     = 0.3              # Lower than fine-tuning scenario
L2_REGULARIZER   = 1e-4             # Slightly tighter weight decay

# ── ArcFace ────────────────────────────────────────────────────────────────
ARCFACE_MARGIN   = 0.5    # Angular margin (~28.6°) — unchanged from paper
ARCFACE_SCALE    = 32.0   # Lowered from 64 → softer temperature for small dataset.
                           # With ~8K training images the original scale=64 makes
                           # the loss overconfident; 32 gives better generalisation.

# ── Training ───────────────────────────────────────────────────────────────
# Single-phase training — no frozen backbone concept.
BATCH_SIZE       = 64     # Larger batch → more stable gradients from scratch.

# Total training budget
TOTAL_EPOCHS     = 120    # 120 epochs at ~74 steps/epoch on LFW ≈ 15 min GPU

# Cosine LR schedule parameters
WARMUP_EPOCHS    = 10     # Linear LR ramp from ~0 to INITIAL_LR
INITIAL_LR       = 1e-2   # Peak LR after warm-up for from-scratch training
MIN_LR           = 1e-6   # Floor for cosine decay

# Kept for backward compatibility with existing imports
WARMUP_LR        = INITIAL_LR
FINETUNE_LR      = 1e-3
FINETUNE_EPOCHS  = TOTAL_EPOCHS - WARMUP_EPOCHS
UNFREEZE_TOP_LAYERS = 0   # Not used in from-scratch training

GRADIENT_CLIP_NORM        = 1.0
EARLY_STOPPING_PATIENCE   = 20   # More patience — from-scratch loss is noisier
REDUCE_LR_PATIENCE        = 7
REDUCE_LR_FACTOR          = 0.5

# ── Mixed precision ────────────────────────────────────────────────────────
# float16 mixed precision: ~2× training speedup on GPUs with Tensor Cores
# (Turing / Ampere: T4, A100, RTX 30xx).  Auto-disabled for older GPUs.
MIXED_PRECISION  = True

# ── Augmentation ───────────────────────────────────────────────────────────
# Aggressive augmentation is especially important when training from scratch
# to prevent overfitting on the small LFW dataset.
AUGMENT_BRIGHTNESS   = 0.25
AUGMENT_CONTRAST     = 0.25
AUGMENT_SATURATION   = 0.15
AUGMENT_HUE          = 0.05
AUGMENT_FLIP         = True
AUGMENT_ROTATION     = 18      # degrees
AUGMENT_ZOOM         = 0.12

# ── Inference / Matching ───────────────────────────────────────────────────
SAME_PERSON_THRESHOLD   = 0.55   # Cosine similarity cutoff for 1:1 verification
FACE_CONF_THRESHOLD     = 0.50   # YOLOv8 minimum detection confidence
MIN_FACE_SIZE           = 20     # Minimum face bounding-box dimension (pixels)

# ── Alignment ──────────────────────────────────────────────────────────────
# Eye landmark indices in YOLOv8-face 5-point keypoints
# Order: [left-eye, right-eye, nose, mouth-left, mouth-right]
LEFT_EYE_IDX      = 0
RIGHT_EYE_IDX     = 1
DESIRED_LEFT_EYE  = (0.35, 0.40)   # Fraction of output image width / height
DESIRED_RIGHT_EYE = (0.65, 0.40)

# ── Evaluation ─────────────────────────────────────────────────────────────
EVAL_PAIRS_PER_CLASS   = 50     # Pairs per identity for verification eval
EER_THRESHOLD_STEPS    = 1000   # Resolution of the EER binary search
NUM_VERIFICATION_PAIRS = 2000   # Pairs used by VerificationCallback during training