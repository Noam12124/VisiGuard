"""
VisiGuard Training
==================
Implements the two-phase transfer-learning training strategy:

  Phase 1  (warm-up)    – backbone frozen, head trained from scratch.
                          High LR; converges the head without corrupting
                          pretrained backbone weights.

  Phase 2  (fine-tune)  – top backbone layers unfrozen, very low LR.
                          Adapts high-level face features while keeping
                          low-level edge detectors intact.

Usage (from CLI)
----------------
    python train.py

The best checkpoint is automatically saved to models/best_model.keras.
"""

import os
import tensorflow as tf

import config
import utils
import model as model_module

logger = utils.get_logger()


def run_training(
    train_ds: tf.data.Dataset,
    val_ds:   tf.data.Dataset,
    num_classes: int,
) -> tuple[tf.keras.Model, dict, dict]:
    """
    Execute both training phases and return the best model + history dicts.

    Parameters
    ----------
    train_ds    : augmented, shuffled training tf.data.Dataset
    val_ds      : validation tf.data.Dataset (no augmentation)
    num_classes : number of identity classes

    Returns
    -------
    best_model   : loaded from checkpoint (best val_accuracy)
    hist1        : History.history dict from phase 1
    hist2        : History.history dict from phase 2
    """
    utils.ensure_dirs()

    # ── Build fresh model ─────────────────────
    m = model_module.build_model(num_classes)
    model_module.print_summary(m)

    # ─────────────────────────────────────────
    # Phase 1: Train head only (backbone frozen)
    # ─────────────────────────────────────────
    logger.info("=" * 55)
    logger.info(f"PHASE 1 — warm-up  (max {config.PHASE1_EPOCHS} epochs)")
    logger.info("=" * 55)

    history1 = m.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.PHASE1_EPOCHS,
        callbacks=model_module.get_callbacks(phase=1),
        verbose=1,
    )

    best_val_acc_p1 = max(history1.history["val_accuracy"])
    logger.info(f"Phase 1 best val accuracy: {best_val_acc_p1:.4f}")

    # ─────────────────────────────────────────
    # Phase 2: Fine-tune top backbone + head
    # ─────────────────────────────────────────
    logger.info("=" * 55)
    logger.info(f"PHASE 2 — fine-tune (max {config.PHASE2_EPOCHS} epochs)")
    logger.info("=" * 55)

    m = model_module.unfreeze_for_phase2(m)

    history2 = m.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.PHASE2_EPOCHS,
        callbacks=model_module.get_callbacks(phase=2),
        verbose=1,
    )

    best_val_acc_p2 = max(history2.history["val_accuracy"])
    logger.info(f"Phase 2 best val accuracy: {best_val_acc_p2:.4f}")

    # ── Load the single best checkpoint ───────
    logger.info(f"Loading best checkpoint from {config.CHECKPOINT_PATH} …")
    best_model = tf.keras.models.load_model(config.CHECKPOINT_PATH)

    overall_best = max(best_val_acc_p1, best_val_acc_p2)
    logger.info(f"Overall best val accuracy: {overall_best:.4f}")

    if overall_best < 0.85:
        logger.warning(
            f"Val accuracy ({overall_best:.4f}) is below the 85% target. "
            "Suggestions: increase PHASE2_EPOCHS, lower MIN_IMAGES_PER_CLASS "
            "to include more classes, or increase UNFREEZE_FROM."
        )
    else:
        logger.info("✓ 85% accuracy target achieved!")

    # ── Plot training curves ───────────────────
    utils.plot_training_curves(history1.history, history2.history)

    return best_model, history1.history, history2.history


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Import here to avoid circular dependency when imported as a module
    from dataset import load_all

    (train_ds, val_ds, test_ds,
     y_test, le, num_classes, all_labels) = load_all()

    best_model, h1, h2 = run_training(train_ds, val_ds, num_classes)

    logger.info("Training complete. Run evaluate.py for full metrics.")
