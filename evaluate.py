"""
VisiGuard Evaluation
====================
Loads the best saved model and generates a comprehensive evaluation report:

  • Test-set accuracy and loss
  • Per-class precision, recall, F1-score
  • Confusion matrix heatmap
  • Training curves (if history files exist)

Usage
-----
    python evaluate.py
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

import config
import utils

logger = utils.get_logger()


# ─────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────

def evaluate(
    model:    tf.keras.Model,
    test_ds:  tf.data.Dataset,
    y_test:   np.ndarray,
    le,                         # fitted sklearn LabelEncoder
) -> dict:
    """
    Run full evaluation on the held-out test set.

    Parameters
    ----------
    model   : trained tf.keras.Model
    test_ds : test tf.data.Dataset (no augmentation, no shuffle)
    y_test  : integer ground-truth labels (numpy array)
    le      : fitted LabelEncoder (for class name lookup)

    Returns
    -------
    metrics : dict with keys 'accuracy', 'loss', 'precision_macro',
              'recall_macro', 'f1_macro'
    """
    utils.ensure_dirs()
    class_names = list(le.classes_)

    # ── 1. Keras built-in loss + accuracy ─────
    logger.info("Running model.evaluate() on test set …")
    loss, acc = model.evaluate(test_ds, verbose=0)
    logger.info(f"Test loss:     {loss:.4f}")
    logger.info(f"Test accuracy: {acc:.4f}  ({acc*100:.2f}%)")

    # ── 2. Collect predictions ─────────────────
    logger.info("Collecting raw predictions …")
    y_pred_probs = model.predict(test_ds, verbose=0)   # (N, num_classes)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    # Sanity check: lengths must match
    if len(y_pred) != len(y_test):
        # test_ds may have dropped last partial batch; align lengths
        n = min(len(y_pred), len(y_test))
        y_pred = y_pred[:n]
        y_test = y_test[:n]

    # ── 3. Classification report ───────────────
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    logger.info("\nClassification Report:\n" + report)

    # Save report to file
    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("VisiGuard – Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Accuracy : {acc*100:.2f}%\n")
        f.write(f"Test Loss     : {loss:.4f}\n\n")
        f.write(report)
    logger.info(f"Classification report saved → {report_path}")

    # ── 4. Confusion matrix ────────────────────
    cm = confusion_matrix(y_test, y_pred)
    utils.plot_confusion_matrix(cm, class_names)

    # ── 5. Aggregate metrics ───────────────────
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy":        float(acc),
        "loss":            float(loss),
        "precision_macro": float(precision),
        "recall_macro":    float(recall),
        "f1_macro":        float(f1),
    }

    logger.info(
        f"\n{'─'*45}\n"
        f"  Accuracy  : {metrics['accuracy']*100:.2f}%\n"
        f"  Precision : {metrics['precision_macro']*100:.2f}%\n"
        f"  Recall    : {metrics['recall_macro']*100:.2f}%\n"
        f"  F1 Score  : {metrics['f1_macro']*100:.2f}%\n"
        f"{'─'*45}"
    )

    # ── 6. Per-class accuracy bar chart ────────
    _plot_per_class_accuracy(y_test, y_pred, class_names)

    return metrics


# ─────────────────────────────────────────────
# Per-class accuracy helper
# ─────────────────────────────────────────────

def _plot_per_class_accuracy(y_true, y_pred, class_names):
    """Bar chart of per-class accuracy (how many of each identity correct)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_class_acc = []
    for cls_idx in range(len(class_names)):
        mask = y_true == cls_idx
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append((y_pred[mask] == cls_idx).mean())

    # Sort by accuracy ascending (worst first) for easy reading
    order = np.argsort(per_class_acc)
    sorted_names = [class_names[i].split("_")[0] for i in order]
    sorted_acc   = [per_class_acc[i] for i in order]

    n = len(class_names)
    fig_w = max(12, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 5), dpi=130)

    colors = ["#EF5350" if a < 0.5 else
              "#FFA726" if a < 0.80 else
              "#66BB6A" for a in sorted_acc]

    ax.bar(range(len(sorted_names)), sorted_acc, color=colors, edgecolor="white")
    ax.set_xticks(range(len(sorted_names)))
    ax.axhline(0.85, color="#1565C0", linestyle="--", linewidth=1.5,
               label="85% target")
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy"); ax.set_title("Per-Identity Accuracy", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(config.RESULTS_DIR, "per_class_accuracy.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Per-class accuracy chart saved → {out}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from dataset import load_all

    (train_ds, val_ds, test_ds,
     y_test, le, num_classes, _) = load_all()

    if not os.path.exists(config.CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No trained model found at {config.CHECKPOINT_PATH}. "
            "Run train.py first."
        )

    logger.info(f"Loading model from {config.CHECKPOINT_PATH} …")
    model = tf.keras.models.load_model(config.CHECKPOINT_PATH)

    metrics = evaluate(model, test_ds, y_test, le)
    logger.info("Evaluation complete. Check results/ directory.")
