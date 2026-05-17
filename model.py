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
# EMBEDDING MODEL (for ArcFace)
# ─────────────────────────────────────────────

def build_model(num_classes: int):

    backbone = _build_backbone()

    inputs = tf.keras.Input(shape=config.IMAGE_SHAPE, name="image")

    # EfficientNet expects 0–255 input
    x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)

    # Feature extraction
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Embedding layer
    x = layers.Dense(
        config.EMBEDDING_DIM,
        kernel_regularizer=regularizers.l2(config.L2_LAMBDA)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # L2-normalized embeddings
    embeddings = layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=1),
        name="embeddings"
    )(x)

    model = Model(inputs, embeddings, name="VisiGuard_Embedding")

    logger.info("Embedding model built successfully")

    # Attach backbone for fine-tuning
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

    # Unfreeze last N layers (except BatchNorm)
    for layer in backbone.layers[start:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    logger.info("Backbone partially unfrozen")

    return model


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def print_summary(model):
    model.summary(expand_nested=False)
