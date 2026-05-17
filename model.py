import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
import config
import utils

logger = utils.get_logger()


# ─────────────────────────────────────────────
# MODEL BUILD
# ─────────────────────────────────────────────

def build_model(num_classes: int) -> tf.keras.Model:

    try:
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=config.IMAGE_SHAPE,
        )
        logger.info("Loaded ImageNet weights.")
    except Exception as e:
        logger.warning(f"Backbone fallback to random init: {e}")
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=config.IMAGE_SHAPE,
        )

    backbone.trainable = False

    inputs = tf.keras.Input(shape=config.IMAGE_SHAPE, name="face_input")

    x = backbone(inputs, training=False)

    # ─────────────────────────────
    # GLOBAL FEATURE EXTRACTION
    # ─────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)

    # ─────────────────────────────
    # EMBEDDING LAYER (IMPORTANT FIX)
    # ─────────────────────────────
    x = layers.Dense(
        config.EMBEDDING_DIM,
        kernel_regularizer=regularizers.l2(config.L2_LAMBDA),
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(config.DROPOUT_1)(x)

    # 🔥 FIX: L2 NORMALIZATION (CRITICAL FOR FACE RECOGNITION)
    x = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1))(x)

    # ─────────────────────────────
    # CLASSIFICATION HEAD
    # ─────────────────────────────
    x = layers.Dense(
        config.DENSE_2_DIM,
        kernel_regularizer=regularizers.l2(config.L2_LAMBDA),
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(config.DROPOUT_2)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
    )(x)

    model = Model(inputs, outputs, name="VisiGuard")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.PHASE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Model built successfully")

    return model


# ─────────────────────────────────────────────
# PHASE 2 UNFREEZE (FIXED STABILITY)
# ─────────────────────────────────────────────

def unfreeze_for_phase2(model: tf.keras.Model) -> tf.keras.Model:

    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True

    # STEP 1: freeze everything
    for layer in backbone.layers:
        layer.trainable = False

    # STEP 2: gradually unfreeze only top layers
    total_layers = len(backbone.layers)
    start = max(0, total_layers + config.UNFREEZE_FROM)

    for layer in backbone.layers[start:]:
        # CRITICAL: keep BatchNorm frozen (prevents accuracy collapse)
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    # STEP 3: VERY LOW LR (stability fix)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.PHASE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info(
        f"Phase 2: unfrozen last {abs(config.UNFREEZE_FROM)} layers"
    )

    return model


# ─────────────────────────────────────────────
# CALLBACKS (STABLE FOR VGGFACE2)
# ─────────────────────────────────────────────

def get_callbacks(phase: int = 1):

    return [
        tf.keras.callbacks.ModelCheckpoint(
            config.CHECKPOINT_PATH,
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


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def print_summary(model):
    model.summary(expand_nested=False, show_trainable=True)