import os
import tensorflow as tf

import config
import utils
import model as model_module

logger = utils.get_logger()


# ─────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────

def run_training(train_ds, val_ds, num_classes):

    utils.ensure_dirs()

    model = model_module.build_model(num_classes)
    model_module.print_summary(model)

    # ─────────────────────────────
    # PHASE 1 (HEAD TRAINING)
    # ─────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1 — Frozen backbone training")
    logger.info("=" * 60)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.PHASE1_EPOCHS,
        callbacks=model_module.get_callbacks(phase=1),
        verbose=1,
    )

    best_p1 = max(history1.history["val_accuracy"])
    logger.info(f"Phase 1 best val accuracy: {best_p1:.4f}")

    # Safety check (important for VGGFace2 stability)
    if best_p1 < 0.30:
        logger.warning(
            "Phase 1 accuracy is very low. "
            "Dataset may be too strict or imbalanced."
        )

    # ─────────────────────────────
    # PHASE 2 (FINE TUNING)
    # ─────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 — Fine-tuning backbone")
    logger.info("=" * 60)

    model = model_module.unfreeze_for_phase2(model)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.PHASE2_EPOCHS,
        callbacks=model_module.get_callbacks(phase=2),
        verbose=1,
    )

    best_p2 = max(history2.history["val_accuracy"])
    logger.info(f"Phase 2 best val accuracy: {best_p2:.4f}")

    # ─────────────────────────────
    # LOAD BEST MODEL (IMPORTANT FIX)
    # ─────────────────────────────
    if os.path.exists(config.CHECKPOINT_PATH):
        best_model = tf.keras.models.load_model(config.CHECKPOINT_PATH)
        logger.info("Loaded best checkpoint model.")
    else:
        best_model = model
        logger.warning("No checkpoint found, using last epoch model.")

    # ─────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────
    overall_best = max(best_p1, best_p2)

    logger.info("=" * 60)
    logger.info(f"OVERALL BEST VALIDATION ACC: {overall_best:.4f}")
    logger.info("=" * 60)

    if overall_best >= 0.85:
        logger.info("✓ TARGET ACHIEVED (85%+)")
    else:
        logger.warning(
            "Below 85% target. Recommended fixes:\n"
            "- reduce MIN_IMAGES_PER_CLASS to 3–5\n"
            "- increase PHASE2_EPOCHS\n"
            "- increase UNFREEZE_FROM (more layers)\n"
            "- ensure VGGFace2 train split is used only"
        )

    # curves
    utils.plot_training_curves(history1.history, history2.history)

    return best_model, history1.history, history2.history


# ─────────────────────────────
# CLI
# ─────────────────────────────

if __name__ == "__main__":
    from dataset import load_all

    train_ds, val_ds, test_ds, y_test, le, num_classes, _ = load_all()

    best_model, h1, h2 = run_training(train_ds, val_ds, num_classes)

    logger.info("Training complete.")