"""
VisiGuard Model Architecture
============================
Two-phase transfer learning strategy:

  Phase 1  –  Backbone FROZEN
               Train only the classification head (fast warm-up).
               High LR; converges quickly without destabilising pretrained weights.

  Phase 2  –  Top N backbone layers UNFROZEN
               Fine-tune backbone + head jointly at a much lower LR.
               Allows the pretrained features to adapt to face images.

Architecture
------------
  EfficientNetB0 (pretrained ImageNet)
      ↓ GlobalAveragePooling2D
      ↓ Dense(512, L2) + BatchNorm + Dropout(0.4)
      ↓ Dense(256, L2) + BatchNorm + Dropout(0.3)
      ↓ Dense(num_classes, softmax)

Why EfficientNetB0?
  • Excellent accuracy-per-parameter ratio
  • Pretrained on 1 000 ImageNet classes → strong low-level + mid-level features
  • Input range [0, 255] – no manual normalisation needed (internal rescaling)
  • Small enough to fine-tune on CPU in reasonable time
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
import config
import utils

logger = utils.get_logger()


# ─────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────

def build_model(num_classes: int) -> tf.keras.Model:
    """
    Construct the full VisiGuard classification model.

    Parameters
    ----------
    num_classes : int   number of identity classes in the dataset

    Returns
    -------
    Compiled tf.keras.Model (phase 1 configuration – backbone frozen).
    """
    # ── Backbone ──────────────────────────────
    # Try pretrained ImageNet weights; fall back gracefully if unavailable
    # (air-gapped machine, CI, or sandbox with restricted network).
    try:
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=config.IMAGE_SHAPE,
        )
        logger.info("EfficientNetB0 pretrained weights: loaded (ImageNet).")
    except Exception as exc:
        logger.warning(
            f"Could not fetch pretrained weights ({type(exc).__name__}). "
            "Falling back to random init. On a machine with internet "
            "access, pretrained weights will give significantly better accuracy."
        )
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=config.IMAGE_SHAPE,
        )
    backbone.trainable = False   # freeze for phase 1
    logger.info(
        f"Backbone: EfficientNetB0 | "
        f"Params: {backbone.count_params():,} (all frozen)"
    )

    # ── Input → backbone ──────────────────────
    inputs = tf.keras.Input(shape=config.IMAGE_SHAPE, name="face_input")
    x = backbone(inputs, training=False)

    # ── Head ──────────────────────────────────
    # GlobalAveragePooling collapses spatial dims → compact feature vector
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Embedding block 1
    x = layers.Dense(
        config.EMBEDDING_DIM,
        kernel_regularizer=regularizers.l2(config.L2_LAMBDA),
        name="embed_dense",
    )(x)
    x = layers.BatchNormalization(name="embed_bn")(x)
    x = layers.Activation("relu", name="embed_relu")(x)
    x = layers.Dropout(config.DROPOUT_1, name="embed_dropout")(x)

    # Embedding block 2
    x = layers.Dense(
        config.DENSE_2_DIM,
        kernel_regularizer=regularizers.l2(config.L2_LAMBDA),
        name="dense2",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.Dropout(config.DROPOUT_2, name="dropout2")(x)

    # Classification output
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",       # explicit float32 for numerical stability
        name="predictions",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="VisiGuard")

    # ── Compile for phase 1 ───────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.PHASE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    total   = model.count_params()
    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    logger.info(
        f"Model built | Total params: {total:,} | "
        f"Trainable (phase 1): {trainable:,}"
    )

    return model


def unfreeze_for_phase2(model: tf.keras.Model) -> tf.keras.Model:
    """
    Unfreeze the last UNFREEZE_FROM layers of the backbone and recompile
    at the lower fine-tuning learning rate.

    The top layers have the most task-specific features (higher-level patterns);
    unfreezing them allows adaptation to face images while keeping low-level
    edge/texture detectors intact (they don't need updating).

    Parameters
    ----------
    model : compiled model from phase 1

    Returns
    -------
    Recompiled model ready for phase 2 training.
    """
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True

    # Freeze all backbone layers first, then selectively unfreeze the top ones
    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze last N layers
    for layer in backbone.layers[config.UNFREEZE_FROM:]:
        # Never unfreeze BatchNorm inside backbone – it can corrupt learned stats
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    newly_trainable = sum(
        tf.size(v).numpy() for v in model.trainable_variables
    )
    logger.info(
        f"Phase 2: unfroze last {abs(config.UNFREEZE_FROM)} backbone layers "
        f"(excl. BN). Trainable params: {newly_trainable:,}"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.PHASE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ─────────────────────────────────────────────
# Callbacks factory
# ─────────────────────────────────────────────

def get_callbacks(phase: int = 1) -> list:
    """
    Return the list of Keras callbacks for a given training phase.

    Callbacks used
    --------------
    ModelCheckpoint  – saves the best model by val_accuracy
    EarlyStopping    – halts training when val_accuracy plateaus
    ReduceLROnPlateau– reduces LR when val_loss stalls
    """
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.CHECKPOINT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.REDUCE_LR_MIN,
            verbose=1,
        ),
    ]

    logger.info(f"Callbacks registered for phase {phase}.")
    return callbacks


# ─────────────────────────────────────────────
# Model summary printer (readable)
# ─────────────────────────────────────────────

def print_summary(model: tf.keras.Model) -> None:
    """Print a concise model summary without backbone internals."""
    model.summary(
        line_length=90,
        expand_nested=False,   # keep backbone collapsed
        show_trainable=True,
    )
