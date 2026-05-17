import argparse
import os
import numpy as np
import tensorflow as tf

import config
import utils

from dataset import load_all
from train import run_training
from evaluate import evaluate
from predict import VisiGuardPredictor

logger = utils.get_logger()


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────

tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main(skip_training: bool = False):

    utils.ensure_dirs()

    logger.info("=" * 60)
    logger.info("VisiGuard – ArcFace Face Recognition System")
    logger.info("=" * 60)

    # ─────────────────────────────
    # DATA
    # ─────────────────────────────
    logger.info("\n[STEP 1] Loading dataset …")

    train_ds, val_ds, test_ds, y_test, le, num_classes, _ = load_all()

    logger.info(f"Classes      : {num_classes}")
    logger.info(f"Test samples : {len(y_test)}")

    # ─────────────────────────────
    # TRAINING
    # ─────────────────────────────
    if not skip_training:

        logger.info("\n[STEP 2] Training model …")

        best_model, h1, h2 = run_training(
            train_ds,
            val_ds,
            num_classes
        )

    else:

        logger.info("\n[STEP 2] Loading model …")

        if not os.path.exists(config.CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"No model found at {config.CHECKPOINT_PATH}"
            )

        best_model = tf.keras.models.load_model(config.CHECKPOINT_PATH)

    # ─────────────────────────────
    # EVALUATION
    # ─────────────────────────────
    logger.info("\n[STEP 3] Evaluating model …")

    metrics = evaluate(best_model, test_ds, y_test, le)

    # ─────────────────────────────
    # INFERENCE DEMO
    # ─────────────────────────────
    logger.info("\n[STEP 4] Inference demo …")

    _run_inference_demo(test_ds, le)

    # ─────────────────────────────
    # SUMMARY
    # ─────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Accuracy : {metrics['accuracy']*100:.2f}%")
    logger.info(f"F1 Score : {metrics['f1_macro']*100:.2f}%")
    logger.info(f"Model    : {config.CHECKPOINT_PATH}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
# FIXED INFERENCE
# ─────────────────────────────────────────────

def _run_inference_demo(test_ds, le):

    if not os.path.exists(config.CHECKPOINT_PATH):
        logger.warning("No checkpoint found.")
        return

    predictor = VisiGuardPredictor()

    # IMPORTANT FIX: avoid batch bias
    test_unbatched = test_ds.unbatch()

    samples = list(test_unbatched.take(5))

    logger.info("\n" + "─" * 55)
    logger.info("Inference Demo")
    logger.info("─" * 55)

    for i, (img, lbl) in enumerate(samples):

        img = img.numpy()
        true_idx = lbl.numpy()

        result = predictor.predict_frame(img)

        true_lbl = le.classes_[true_idx]
        pred_lbl = result["identity"]
        conf = result["confidence"]

        mark = "✓" if true_lbl == pred_lbl else "✗"

        logger.info(
            f"[{i+1}] True: {true_lbl:<25} "
            f"| Pred: {pred_lbl:<25} "
            f"({conf*100:.1f}%) {mark}"
        )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(skip_training=args.skip_train)