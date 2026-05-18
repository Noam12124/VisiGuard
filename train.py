import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import config
import utils
from model import build_model, unfreeze_for_phase2
from arcface import ArcFace

logger = utils.get_logger()


# ─────────────────────────────────────────────
# STEP FUNCTIONS (With Accuracy Metric Extraction)
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

    # 🔥 FIX: Compute step categorical accuracy to track runtime performance
    preds = tf.argmax(logits, axis=1, output_type=labels.dtype)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

    return loss, acc


def val_step(model, arcface, images, labels):
    embeddings = model(images, training=False)
    logits = arcface([embeddings, labels])

    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
    )

    # 🔥 FIX: Compute validation step categorical accuracy
    preds = tf.argmax(logits, axis=1, output_type=labels.dtype)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

    return loss, acc


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

    # 🔥 CRITICAL FIX: Pass configured margin and scale arguments explicitly.
    # Leaving this empty fell back to ArcFace layer defaults (0.5, 64.0), 
    # which caused massive gradient explosions and flattened accuracy.
    arcface = ArcFace(num_classes=num_classes, margin=config.ARC_MARGIN, scale=config.ARC_SCALE)

    # Dictionary to track training history records smoothly
    history_p1 = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    history_p2 = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    # ─────────────────────────────
    # PHASE 1 - Frozen Backbone Training
    # ─────────────────────────────
    optimizer = tf.keras.optimizers.Adam(config.PHASE1_LR)
    best_val = float("inf")

    logger.info("PHASE 1 - training")

    for epoch in range(config.PHASE1_EPOCHS):
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # TRAIN WITH PROGRESS BAR
        for images, labels in tqdm(train_ds, desc=f"Epoch {epoch+1} Training"):
            loss, acc = train_step(model, arcface, images, labels, optimizer)
            train_losses.append(float(loss))
            train_accs.append(float(acc))

        # VALIDATION WITH PROGRESS BAR
        for images, labels in tqdm(val_ds, desc=f"Epoch {epoch+1} Validation"):
            loss, acc = val_step(model, arcface, images, labels)
            val_losses.append(float(loss))
            val_accs.append(float(acc))

        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)
        val_loss = sum(val_losses) / len(val_losses)
        val_acc = sum(val_accs) / len(val_accs)

        history_p1["loss"].append(train_loss)
        history_p1["acc"].append(train_acc)
        history_p1["val_loss"].append(val_loss)
        history_p1["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch+1}/{config.PHASE1_EPOCHS} "
            f"| train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}% "
            f"| val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%"
        )

        # Save best model if validation loss improves
        if val_loss < best_val:
            best_val = val_loss

            # Save embedding model
            model.save(config.CHECKPOINT_PATH)

            # Save ArcFace weights
            weights_out = os.path.join(config.MODEL_DIR, "arcface_weights.npy")
            np.save(weights_out, arcface.W.numpy())

            logger.info("Saved best model + ArcFace weights")


    # ─────────────────────────────
    # PHASE 2 — Fine‑tuning
    # ─────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 - fine tuning")
    logger.info("=" * 60)

    model = unfreeze_for_phase2(model)
    optimizer = tf.keras.optimizers.Adam(config.PHASE2_LR)

    # Reset best validation target for fine-tuning tracking
    best_val = float("inf")

    for epoch in range(config.PHASE2_EPOCHS):
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # TRAIN WITH PROGRESS BAR
        for images, labels in tqdm(train_ds, desc=f"[FT] Epoch {epoch+1} Training"):
            loss, acc = train_step(model, arcface, images, labels, optimizer)
            train_losses.append(float(loss))
            train_accs.append(float(acc))

        # VALIDATION WITH PROGRESS BAR
        for images, labels in tqdm(val_ds, desc=f"[FT] Epoch {epoch+1} Validation"):
            loss, acc = val_step(model, arcface, images, labels)
            val_losses.append(float(loss))
            val_accs.append(float(acc))

        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)
        val_loss = sum(val_losses) / len(val_losses)
        val_acc = sum(val_accs) / len(val_accs)

        history_p2["loss"].append(train_loss)
        history_p2["acc"].append(train_acc)
        history_p2["val_loss"].append(val_loss)
        history_p2["val_acc"].append(val_acc)

        logger.info(
            f"[FT] Epoch {epoch+1}/{config.PHASE2_EPOCHS} "
            f"| train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}% "
            f"| val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%"
        )

        # CHECKPOINT EVALUATION INSIDE THE LOOP
        if val_loss < best_val:
            best_val = val_loss

            # Save embedding model
            model.save(config.CHECKPOINT_PATH)

            # Save ArcFace classifier weights explicitly using clean os.path joins
            weights_out = os.path.join(config.MODEL_DIR, "arcface_weights.npy")
            np.save(weights_out, arcface.W.numpy())

            logger.info("Saved best fine-tuned model + ArcFace weights")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    # 🔥 FIX: Return histories instead of redundant empty placeholders
    return model, history_p1, history_p2