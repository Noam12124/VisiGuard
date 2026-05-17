"""
VisiGuard – Main Entry Point
=============================
Runs the complete pipeline end-to-end:

  1. Dataset  – download, scan, filter, encode, split, build tf.data pipelines
  2. Training – two-phase transfer learning (warm-up → fine-tune)
  3. Evaluation – accuracy, confusion matrix, per-class F1
  4. Inference demo – predict on a random test sample

Usage
-----
    python main.py              # full pipeline
    python main.py --skip-train # load existing model, run eval only
"""

import argparse
import os
import numpy as np
import tensorflow as tf

import config
import utils
from dataset  import load_all
from train    import run_training
from evaluate import evaluate
from predict  import VisiGuardPredictor

logger = utils.get_logger()


def main(skip_training: bool = False) -> None:
    """
    Orchestrate the full VisiGuard pipeline.

    Parameters
    ----------
    skip_training : if True, skip training and load the existing checkpoint.
    """
    utils.ensure_dirs()

    logger.info("=" * 60)
    logger.info("  VisiGuard – Face Recognition System")
    logger.info("=" * 60)

    # ── Step 1: Data ──────────────────────────────────────────────
    logger.info("\n[STEP 1] Building dataset pipeline …")
    (train_ds, val_ds, test_ds,
     y_test, le, num_classes, all_labels) = load_all()

    logger.info(f"  Classes       : {num_classes}")
    logger.info(f"  Test samples  : {len(y_test)}")

    # ── Step 2: Training ──────────────────────────────────────────
    if not skip_training:
        logger.info("\n[STEP 2] Training model …")
        best_model, h1, h2 = run_training(train_ds, val_ds, num_classes)
    else:
        logger.info("\n[STEP 2] Skipping training – loading checkpoint …")
        if not os.path.exists(config.CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"No model at {config.CHECKPOINT_PATH}. "
                "Run without --skip-train first."
            )
        best_model = tf.keras.models.load_model(config.CHECKPOINT_PATH)

    # ── Step 3: Evaluation ────────────────────────────────────────
    logger.info("\n[STEP 3] Evaluating on test set …")
    metrics = evaluate(best_model, test_ds, y_test, le)

    # ── Step 4: Inference demo ────────────────────────────────────
    logger.info("\n[STEP 4] Running inference demo on 5 random test samples …")
    _run_inference_demo(train_ds, test_ds, y_test, le)

    # ── Summary ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Test Accuracy  : {metrics['accuracy']*100:.2f}%")
    logger.info(f"  Precision (M)  : {metrics['precision_macro']*100:.2f}%")
    logger.info(f"  Recall (M)     : {metrics['recall_macro']*100:.2f}%")
    logger.info(f"  F1 Score (M)   : {metrics['f1_macro']*100:.2f}%")
    logger.info(f"  Model saved    : {config.CHECKPOINT_PATH}")
    logger.info(f"  Plots saved    : {config.RESULTS_DIR}/")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
# Inference demo helper
# ─────────────────────────────────────────────

def _run_inference_demo(train_ds, test_ds, y_test, le) -> None:
    """
    Sample 5 images from the test set, run inference, and log results.
    This verifies that the VisiGuardPredictor pipeline works end-to-end.
    """
    if not os.path.exists(config.CHECKPOINT_PATH):
        logger.warning("No checkpoint found; skipping inference demo.")
        return

    predictor = VisiGuardPredictor()

    # Collect a few raw images from the test pipeline
    demo_images, demo_labels = [], []
    for batch_imgs, batch_lbls in test_ds.take(2):
        demo_images.append(batch_imgs.numpy())
        demo_labels.append(batch_lbls.numpy())

    demo_images = np.concatenate(demo_images, axis=0)[:5]
    demo_labels = np.concatenate(demo_labels, axis=0)[:5]

    logger.info(f"\n{'─'*55}")
    logger.info("  Inference Demo (5 random test samples)")
    logger.info(f"{'─'*55}")

    for i, (img, true_idx) in enumerate(zip(demo_images, demo_labels)):
        # Convert float32 [0,255] → uint8 BGR for the predictor helper
        import cv2
        rgb  = np.clip(img, 0, 255).astype(np.uint8)
        bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        result   = predictor.predict_frame(bgr)
        true_lbl = le.classes_[true_idx]
        pred_lbl = result["identity"]
        conf     = result["confidence"]
        correct  = "✓" if pred_lbl == true_lbl else "✗"

        logger.info(
            f"  [{i+1}] True: {true_lbl:<25} | "
            f"Pred: {pred_lbl:<25} ({conf*100:.1f}%)  {correct}"
        )

    logger.info(f"{'─'*55}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="VisiGuard – Face Recognition Pipeline"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; load existing checkpoint and evaluate only."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(skip_training=args.skip_train)
