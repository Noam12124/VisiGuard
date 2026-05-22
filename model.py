"""
model.py — Face recognition CNN.

Architecture:
    Input (112×112×3)
        │
        ▼
    EfficientNetV2-S  [ImageNet pretrained]
        │
        ▼
    GlobalAveragePooling2D  → (1280,)
    BatchNormalization
        │
        ▼
    Dense(1024, no bias)  +  BatchNorm  +  PReLU  +  Dropout(0.4)
        │
        ▼
    Dense(512, no bias, L2)  +  BatchNorm
    L2-Normalise  ────────────────────  Embedding (512,) for inference
        │
        ▼  [training only]
    ArcFaceLayer  →  Logits (num_classes,)

Why EfficientNetV2-S?
  • Stronger feature extractor than ResNet50V2 at the same FLOPs.
  • Fused-MBConv blocks train faster and generalise better.
  • Pretrained on ImageNet-21k (via the -S variant in TF) → better init.
  • 112×112 input is within its efficient operating range.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import (
    EfficientNetV2S,
    ResNet50V2,
)
import config
from arcface import ArcFaceLayer


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_backbone(name: str, input_shape: tuple):
    """Return a pretrained backbone with top removed."""
    kwargs = dict(
        include_top    = False,
        weights        = "imagenet",
        input_shape    = input_shape,
        pooling        = None,
    )
    name = name.lower()
    if name == "efficientnetv2s":
        return EfficientNetV2S(include_preprocessing=False, **kwargs)
    elif name == "resnet50v2":
        return ResNet50V2(**kwargs)
    elif name == "convnext_tiny":
        try:
            from tensorflow.keras.applications import ConvNeXtTiny
            return ConvNeXtTiny(**kwargs)
        except ImportError:
            print("[model] ConvNeXtTiny not available; falling back to EfficientNetV2S.")
            return EfficientNetV2S(include_preprocessing=False, **kwargs)
    else:
        raise ValueError(f"Unknown backbone: {name}")


def _l2_norm(x, name="embedding"):
    return tf.math.l2_normalize(x, axis=1, name=name)


# ── Main builder ───────────────────────────────────────────────────────────

def build_model(num_classes: int, training: bool = True):
    """
    Build the complete model.

    Args:
        num_classes: Number of identities in the training set.
        training:    If True, ArcFace head is attached (for training).
                     If False, the model outputs only the L2-normalised
                     512-dim embedding (for inference).

    Returns:
        (full_model, embedding_model)
          full_model      – has ArcFace logits output (use for training).
          embedding_model – outputs the 512-d embedding (use for inference).

    The embedding_model shares all weights with full_model; you never
    need to copy weights between them.
    """
    input_shape = (*config.IMAGE_SIZE, config.NUM_CHANNELS)

    # ── Input ──────────────────────────────────────────────────────────────
    img_input  = layers.Input(shape=input_shape, name="image_input")
    label_input = layers.Input(shape=(), name="label_input", dtype=tf.int32)

    # ── Preprocessing (backbone-specific normalisation) ────────────────────
    # EfficientNetV2 expects pixels in [0, 255]; we normalise to [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(img_input)

    # ── Backbone ───────────────────────────────────────────────────────────
    backbone = _get_backbone(config.BACKBONE, input_shape)
    backbone.trainable = False   # Phase 1: frozen
    x = backbone(x, training=False)

    # ── Pooling + first BN ────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_gap")(x)

    # ── Bottleneck head ────────────────────────────────────────────────────
    x = layers.Dense(
        1024,
        use_bias    = False,
        kernel_regularizer = regularizers.l2(config.L2_REGULARIZER),
        name        = "dense_1024",
    )(x)
    x = layers.BatchNormalization(name="bn_1024")(x)
    x = layers.PReLU(shared_axes=[1], name="prelu")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="dropout")(x)

    # ── Embedding projection ───────────────────────────────────────────────
    x = layers.Dense(
        config.EMBEDDING_DIM,
        use_bias    = False,
        kernel_regularizer = regularizers.l2(config.L2_REGULARIZER),
        name        = "dense_512",
    )(x)
    x = layers.BatchNormalization(name="bn_512")(x)

    # L2-normalise → unit-sphere embeddings
    embedding = layers.UnitNormalization(axis=1, name="embedding")(x)

    # ── Embedding-only model (inference) ──────────────────────────────────
    embedding_model = models.Model(
        inputs  = img_input,
        outputs = embedding,
        name    = "embedding_model",
    )

    if not training:
        return None, embedding_model

    # ── ArcFace head (training only) ──────────────────────────────────────
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


def freeze_backbone(model):
    """Freeze the backbone (Phase 1)."""
    for layer in model.layers:
        if hasattr(layer, "layers"):   # it's a sub-model (the backbone)
            layer.trainable = False
    print("[model] Backbone frozen.")


def unfreeze_top_layers(model, n: int = config.UNFREEZE_TOP_LAYERS):
    """
    Unfreeze the top `n` layers of the backbone (Phase 2).
    BatchNorm layers inside the backbone remain in inference mode to
    preserve stable statistics.
    """
    backbone = None
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 10:
            backbone = layer
            break

    if backbone is None:
        print("[model] WARNING: could not find backbone sub-model to unfreeze.")
        return

    backbone.trainable = True
    total = len(backbone.layers)
    freeze_until = max(0, total - n)
    for i, layer in enumerate(backbone.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            # Keep BatchNorm in inference mode — critical for stable fine-tuning.
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    print(f"[model] Unfroze top {trainable_count} backbone layers "
          f"(out of {total}; BN layers kept frozen).")


def get_cosine_scheduler(
    base_lr: float,
    total_epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = config.MIN_LR,
):
    """
    Cosine annealing LR schedule with optional linear warmup.

    Returns a tf.keras.callbacks.LearningRateScheduler-compatible function.
    """
    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        import math
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    return schedule


def compile_model(model, lr: float, clip_norm: float = config.GRADIENT_CLIP_NORM):
    """Compile model with Adam + gradient clipping + sparse CE loss."""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = lr,
        clipnorm      = clip_norm,
    )
    model.compile(
        optimizer  = optimizer,
        loss       = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics    = ["accuracy"],
    )
    return model


def model_summary(model):
    """Print summary and count trainable parameters."""
    model.summary(line_length=100)
    trainable     = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    print(f"\nTrainable params:     {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")
    print(f"Total params:         {trainable + non_trainable:,}")