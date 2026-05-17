import tensorflow as tf
import config
import utils

from model import build_model, unfreeze_for_phase2

logger = utils.get_logger()


# ─────────────────────────────────────────────
# STEP FUNCTIONS
# ─────────────────────────────────────────────

def train_step(model, images, labels, optimizer):

    with tf.GradientTape() as tape:
        logits = model([images, labels], training=True)

        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True
            )
        )

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def val_step(model, images, labels):

    logits = model([images, labels], training=False)

    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
    )

    return loss


# ─────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────

def run_training(train_ds, val_ds, num_classes):

    utils.ensure_dirs()

    logger.info("=" * 60)
    logger.info("ARC FACE TRAINING - FINAL CLEAN VERSION")
    logger.info("=" * 60)

    model = build_model(num_classes)

    # ─────────────────────────────
    # PHASE 1
    # ─────────────────────────────
    optimizer = tf.keras.optimizers.Adam(config.PHASE1_LR)

    best_val = float("inf")

    logger.info("PHASE 1 - training")

    for epoch in range(config.PHASE1_EPOCHS):

        train_losses = []
        val_losses = []

        for images, labels in train_ds:
            loss = train_step(model, images, labels, optimizer)
            train_losses.append(float(loss))

        for images, labels in val_ds:
            loss = val_step(model, images, labels)
            val_losses.append(float(loss))

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        logger.info(
            f"Epoch {epoch+1}/{config.PHASE1_EPOCHS} "
            f"| train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            model.save(config.CHECKPOINT_PATH)
            logger.info("Saved best model")

    # ─────────────────────────────
    # PHASE 2
    # ─────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 - fine tuning")
    logger.info("=" * 60)

    model = unfreeze_for_phase2(model)
    optimizer = tf.keras.optimizers.Adam(config.PHASE2_LR)

    best_val = float("inf")

    for epoch in range(config.PHASE2_EPOCHS):

        train_losses = []
        val_losses = []

        for images, labels in train_ds:
            loss = train_step(model, images, labels, optimizer)
            train_losses.append(float(loss))

        for images, labels in val_ds:
            loss = val_step(model, images, labels)
            val_losses.append(float(loss))

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        logger.info(
            f"[FT] Epoch {epoch+1}/{config.PHASE2_EPOCHS} "
            f"| train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            model.save(config.CHECKPOINT_PATH)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    return model, None, None