import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
import config
import utils

logger = utils.get_logger()


# ─────────────────────────────────────────────
# BACKBONE
# ─────────────────────────────────────────────

def _build_backbone():
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=config.IMAGE_SHAPE,
    )

    backbone.trainable = False
    return backbone


# ─────────────────────────────────────────────
# EMBEDDING MODEL (ArcFace-ready)
# ─────────────────────────────────────────────

def build_model(num_classes: int):

    backbone = _build_backbone()

    inputs = tf.keras.Input(shape=config.IMAGE_SHAPE, name="image")

    # EfficientNet expects 0–255 scale inputs explicitly processed
    x = tf.keras.layers.Lambda(
        lambda t: tf.keras.applications.efficientnet.preprocess_input(
            t * 255.0
        ),
        name="preprocess"
    )(inputs)

    # 🔥 CRITICAL BREAKTHROUGH FIX: Removed training=False hardcoding.
    # Leaving this flag out allows Keras to automatically inject the execution loop context.
    # It will safely keep BatchNorm frozen during Phase 1, and permit parameter updates during Phase 2.
    x = backbone(x)
    x = layers.GlobalAveragePooling2D()(x)

    # ─────────────────────────────────────────────
    # EMBEDDING HEAD (Optimized for clean vector separation)
    # ─────────────────────────────────────────────
    x = layers.Dense(
        config.EMBEDDING_DIM,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(config.L2_LAMBDA)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    # L2 normalize embeddings (essential for hyperspherical margin separation)
    embeddings = layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=1),
        name="embeddings"
    )(x)

    model = Model(inputs, embeddings, name="VisiGuard_Embedding")

    logger.info("Embedding model built successfully")

    # Attach backbone for tracking in phase unfreezing
    model.backbone = backbone

    return model


# ─────────────────────────────────────────────
# UNFREEZE FOR PHASE 2
# ─────────────────────────────────────────────

def unfreeze_for_phase2(model):

    backbone = getattr(model, "backbone", None)

    if backbone is None:
        raise ValueError("Backbone not attached")

    backbone.trainable = True

    # Freeze all layers first
    for layer in backbone.layers:
        layer.trainable = False

    total = len(backbone.layers)
    start = max(0, total + config.UNFREEZE_FROM)

    # Unfreeze the deep blocks (except global BatchNorm tracking parameters)
    for layer in backbone.layers[start:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    logger.info(f"Backbone partially unfrozen. Layers from index {start} to {total} are active.")

    return model


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def print_summary(model):
    model.summary(expand_nested=False)