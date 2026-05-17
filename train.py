import tensorflow as tf
from tqdm import tqdm
import config
import utils

from model import build_model, unfreeze_for_phase2
from arcface import ArcFace

logger = utils.get_logger()


# ─────────────────────────────────────────────
# STEP FUNCTIONS
# ─────────────────────────────────────────────

def train_step(model, arcface, images, labels, optimizer):

    with tf.GradientTape() as tape:

        # 1) Get embeddings from the model
        embeddings = model(images, training=True)

        # 2) ArcFace logits
        logits = arcface([embeddings, labels])

        # 3) Compute loss
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True
            )
        )

    # 4) Compute gradients for BOTH model + ArcFace
    vars_all = model.trainable_variables + arcface.trainable_variables
    grads = tape.gradient(loss, vars_all)

    # 5) Apply gradients
    optimizer.apply_gradients(zip(grads, vars_all))

    return loss


def val_step(model, arcface, images, labels):

    embeddings = model(images, training=False)
    logits = arcface([embeddings, labels])

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

    # Build embedding model
    model = build_model(num_classes)

    # Build ArcFace head
    arcface = ArcFace(num_classes)

    # ─────────────────────────────
    # PHASE 1
    # ─────────────────────────────
    optimizer = tf.keras.optimizers.Adam(config.PHASE1_LR)

    best_val = float("inf")

    logger.info("PHASE 1 - training")

    for epoch in range(config.PHASE1_EPOCHS):

        train_losses = []
        val_losses = []

        # TRAIN WITH PROGRESS BAR
        for images, labels in tqdm(train_ds, desc=f"Epoch {epoch+1} Training"):
            loss = train_step(model, arcface, images, labels, optimizer)
            train_losses.append(float(loss))

        # VALIDATION WITH PROGRESS BAR
        for images, labels in tqdm(val_ds, desc=f"Epoch {epoch+1} Validation"):
            loss = val_step(model, arcface, images, labels)
            val_losses.append(float(loss))

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        logger.info(
            f"Epoch {epoch+1}/{config.PHASE1_EPOCHS} "
            f"| train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            model.save(config.CHECKPOINT_PATH)
            logger.info("Saved best model")


    # ─────────────────────────────
    # PHASE 2 — Fine‑tuning
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

        # TRAIN WITH PROGRESS BAR
        for images, labels in tqdm(train_ds, desc=f"[FT] Epoch {epoch+1} Training"):
            loss = train_step(model, arcface, images, labels, optimizer)
            train_losses.append(float(loss))

        # VALIDATION WITH PROGRESS BAR
        for images, labels in tqdm(val_ds, desc=f"[FT] Epoch {epoch+1} Validation"):
            loss = val_step(model, arcface, images, labels)
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
            logger.info("Saved best fine‑tuned model")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    return model, None, None
