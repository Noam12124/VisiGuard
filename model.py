"""
model.py — Face recognition CNN trained entirely from scratch.

Architecture: FaceResNet — a lightweight residual network designed
specifically for 112×112 face images without any pretrained weights.

Why train from scratch?
  • Full control over what features the network learns for faces.
  • No dependency on ImageNet-pretrained checkpoints.
  • Architecture is sized and regularised for the face domain directly.

Architecture overview:
    Input (112×112×3)
        │
        ▼
    Stem: Conv(64, 3×3, stride=2) → BN → PReLU   → (56×56×64)
        │
        ▼
    Stage 1: 2 × ResBlock(64,  stride=1)           → (56×56×64)
    Stage 2: 2 × ResBlock(128, first stride=2)     → (28×28×128)
    Stage 3: 4 × ResBlock(256, first stride=2)     → (14×14×256)
    Stage 4: 2 × ResBlock(512, first stride=2)     → (7×7×512)
        │
        ▼
    GlobalAveragePooling2D → (512,)
    BatchNormalization
        │
        ▼
    Dense(1024, no bias) + BN + PReLU + Dropout(0.3)
        │
        ▼
    Dense(512, no bias) + BN
    L2-Normalise  ───────────────────  Embedding (512,) for inference
        │
        ▼  [training only]
    ArcFaceLayer  →  Logits (num_classes,)

Expected accuracy on LFW (~1,680 identities, ≥5 images each):
    Validation accuracy: ~85% after 120 epochs on GPU.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import config
from arcface import ArcFaceLayer


# ── Residual Block ─────────────────────────────────────────────────────────

def _residual_block(x, filters: int, stride: int = 1, name: str = "rb"):
    """
    Basic residual block:
        x → Conv(3×3) → BN → PReLU → Conv(3×3) → BN → Add(shortcut) → PReLU

    The shortcut connection uses a 1×1 conv + BN whenever the spatial size
    or channel count changes so dimensions always match.

    He (kaiming) initialisation is used throughout — essential for deep
    networks trained from scratch to avoid vanishing/exploding gradients.
    """
    shortcut = x

    # ── First conv ────────────────────────────────────────────────────────
    x = layers.Conv2D(
        filters,
        kernel_size   = 3,
        strides       = stride,
        padding       = "same",
        use_bias      = False,
        kernel_initializer  = "he_normal",
        kernel_regularizer  = regularizers.l2(config.L2_REGULARIZER),
        name          = f"{name}_c1",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.PReLU(shared_axes=[1, 2], name=f"{name}_pr1")(x)

    # ── Second conv ───────────────────────────────────────────────────────
    x = layers.Conv2D(
        filters,
        kernel_size   = 3,
        padding       = "same",
        use_bias      = False,
        kernel_initializer  = "he_normal",
        kernel_regularizer  = regularizers.l2(config.L2_REGULARIZER),
        name          = f"{name}_c2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    # ── Shortcut projection (only when shape changes) ─────────────────────
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            kernel_size  = 1,
            strides      = stride,
            use_bias     = False,
            kernel_initializer = "he_normal",
            name         = f"{name}_sc",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_sc_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.PReLU(shared_axes=[1, 2], name=f"{name}_pr2")(x)
    return x


# ── Residual Stage Helper ──────────────────────────────────────────────────

def _residual_stage(x, filters: int, num_blocks: int, first_stride: int = 2,
                    stage_name: str = "s"):
    """
    Stack `num_blocks` residual blocks.  The first block uses `first_stride`
    (typically 2 for downsampling); the rest use stride=1.
    """
    for i in range(num_blocks):
        stride = first_stride if i == 0 else 1
        x = _residual_block(x, filters, stride=stride,
                             name=f"{stage_name}_b{i + 1}")
    return x


# ── Main Model Builder ────────────────────────────────────────────────────

def build_model(num_classes: int, training: bool = True):
    """
    Build the FaceResNet model.

    Args:
        num_classes: Number of identities in the training set.
        training:    If True, ArcFace head is attached for training.
                     If False, only the 512-d embedding model is returned.

    Returns:
        (full_model, embedding_model)
          full_model      – ArcFace logit output (use for training).
          embedding_model – 512-d L2-normalised embedding (use for inference).

    Both models share weights; no need to copy anything between them.
    """
    input_shape = (*config.IMAGE_SIZE, config.NUM_CHANNELS)  # (112, 112, 3)

    # ── Inputs ──────────────────────────────────────────────────────────
    img_input   = layers.Input(shape=input_shape, name="image_input")
    label_input = layers.Input(shape=(), name="label_input", dtype=tf.int32)

    # ── Normalise pixels to [-1, 1] ───────────────────────────────────
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(img_input)

    # ── Stem: aggressive first downsample, stride=2 ───────────────────
    # 112×112 → 56×56
    x = layers.Conv2D(
        64,
        kernel_size  = 3,
        strides      = 2,
        padding      = "same",
        use_bias     = False,
        kernel_initializer = "he_normal",
        kernel_regularizer = regularizers.l2(config.L2_REGULARIZER),
        name         = "stem_conv",
    )(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.PReLU(shared_axes=[1, 2], name="stem_prelu")(x)

    # ── Residual stages ──────────────────────────────────────────────
    # Stage 1: 56×56 × 64  (no spatial downsampling)
    x = _residual_stage(x, filters=64,  num_blocks=2, first_stride=1, stage_name="s1")

    # Stage 2: 56×56 → 28×28 × 128
    x = _residual_stage(x, filters=128, num_blocks=2, first_stride=2, stage_name="s2")

    # Stage 3: 28×28 → 14×14 × 256
    x = _residual_stage(x, filters=256, num_blocks=4, first_stride=2, stage_name="s3")

    # Stage 4: 14×14 → 7×7 × 512
    x = _residual_stage(x, filters=512, num_blocks=2, first_stride=2, stage_name="s4")

    # ── Embedding head ────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_gap")(x)

    x = layers.Dense(
        1024,
        use_bias           = False,
        kernel_initializer = "he_normal",
        kernel_regularizer = regularizers.l2(config.L2_REGULARIZER),
        name               = "dense_1024",
    )(x)
    x = layers.BatchNormalization(name="bn_1024")(x)
    x = layers.PReLU(shared_axes=[1], name="prelu_head")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="dropout")(x)

    x = layers.Dense(
        config.EMBEDDING_DIM,
        use_bias           = False,
        kernel_initializer = "he_normal",
        kernel_regularizer = regularizers.l2(config.L2_REGULARIZER),
        name               = "dense_512",
    )(x)
    x = layers.BatchNormalization(name="bn_512")(x)

    # L2-normalise → embeddings lie on the unit hypersphere
    embedding = layers.UnitNormalization(axis=1, name="embedding")(x)

    # ── Inference model ───────────────────────────────────────────────
    embedding_model = models.Model(
        inputs  = img_input,
        outputs = embedding,
        name    = "embedding_model",
    )

    if not training:
        return None, embedding_model

    # ── ArcFace head (training only) ──────────────────────────────────
    arcface_layer = ArcFaceLayer(
        num_classes = num_classes,
        margin      = config.ARCFACE_MARGIN,
        scale       = config.ARCFACE_SCALE,
        name        = "arcface",
    )
    logits = arcface_layer(embedding, labels=label_input, training=True)

    full_model = models.Model(
        inputs  = [img_input, label_input],
        outputs = logits,
        name    = "face_recognition_model",
    )

    return full_model, embedding_model


# ── No-op stubs kept for API compatibility ────────────────────────────────
# (These were used in the pretrained two-phase training setup.
#  Since this model is trained fully from scratch in a single phase,
#  they are no longer needed but kept so any existing import does not break.)

def freeze_backbone(model):
    """No-op: there is no pretrained backbone to freeze."""
    print("[model] freeze_backbone: no-op — model trains fully from scratch.")


def unfreeze_top_layers(model, n: int = None):
    """No-op: there is no pretrained backbone to unfreeze."""
    print("[model] unfreeze_top_layers: no-op — model trains fully from scratch.")


# ── LR Scheduler ──────────────────────────────────────────────────────────

def get_cosine_scheduler(
    base_lr:       float,
    total_epochs:  int,
    warmup_epochs: int = 0,
    min_lr:        float = config.MIN_LR,
):
    """
    Cosine annealing LR schedule with optional linear warm-up.

    Warm-up is crucial when training from scratch: it ramps the LR
    from near-zero up to `base_lr` over the first `warmup_epochs`
    epochs so the random initial weights don't produce huge gradient
    spikes.

    Returns a Keras LearningRateScheduler-compatible callable.
    """
    def schedule(epoch: int, lr: float) -> float:
        import math
        if epoch < warmup_epochs:
            # Linear ramp: epoch 0 → base_lr/warmup, epoch W-1 → base_lr
            return base_lr * (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    return schedule


# ── Compile helper ────────────────────────────────────────────────────────

def compile_model(model, lr: float, clip_norm: float = config.GRADIENT_CLIP_NORM):
    """Compile with Adam + gradient clipping + Sparse Categorical Cross-Entropy."""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = lr,
        clipnorm      = clip_norm,
    )
    model.compile(
        optimizer = optimizer,
        loss      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics   = ["accuracy"],
    )
    return model


# ── Summary helper ────────────────────────────────────────────────────────

def model_summary(model):
    """Print layer summary and total parameter count."""
    model.summary(line_length=100)
    trainable     = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    print(f"\nTrainable params:     {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")
    print(f"Total params:         {trainable + non_trainable:,}")