import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config
import utils

logger = utils.get_logger()


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, test_ds, y_test, le):

    utils.ensure_dirs()

    class_names = list(le.classes_)

    # ─────────────────────────────
    # 1. Loss + accuracy
    # ─────────────────────────────
    logger.info("Evaluating model …")
    loss, acc = model.evaluate(test_ds, verbose=0)

    logger.info(f"Test loss: {loss:.4f}")
    logger.info(f"Test accuracy: {acc*100:.2f}%")

    # ─────────────────────────────
    # 2. Predictions (ORDER SAFE FIX)
    # ─────────────────────────────
    logger.info("Generating predictions …")

    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_test = np.array(y_test)

    # FIX: strict alignment instead of silent truncation
    min_len = min(len(y_pred), len(y_test))

    if len(y_pred) != len(y_test):
        logger.warning(
            f"Length mismatch detected → "
            f"pred={len(y_pred)} test={len(y_test)} → trimming to {min_len}"
        )

    y_pred = y_pred[:min_len]
    y_test = y_test[:min_len]

    # ─────────────────────────────
    # 3. Classification report
    # ─────────────────────────────
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names[:len(np.unique(y_test))],
        digits=4,
        zero_division=0,
    )

    logger.info("\n" + report)

    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Saved report → {report_path}")

    # ─────────────────────────────
    # 4. Confusion matrix
    # ─────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    utils.plot_confusion_matrix(cm, class_names)

    # ─────────────────────────────
    # 5. Macro metrics
    # ─────────────────────────────
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(acc),
        "loss": float(loss),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }

    logger.info(
        f"\n──────── RESULTS ────────\n"
        f"Accuracy  : {acc*100:.2f}%\n"
        f"Precision : {precision*100:.2f}%\n"
        f"Recall    : {recall*100:.2f}%\n"
        f"F1        : {f1*100:.2f}%\n"
        f"────────────────────────"
    )

    # ─────────────────────────────
    # 6. Per-class accuracy
    # ─────────────────────────────
    _plot_per_class_accuracy(y_test, y_pred, class_names)

    return metrics


# ─────────────────────────────────────────────
# PER CLASS PLOT (FIXED FOR VGGFACE2)
# ─────────────────────────────────────────────

def _plot_per_class_accuracy(y_true, y_pred, class_names):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_class_acc = []

    for i in range(len(class_names)):
        mask = y_true == i

        if np.sum(mask) == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(np.mean(y_pred[mask] == i))

    order = np.argsort(per_class_acc)

    sorted_acc = [per_class_acc[i] for i in order]
    sorted_names = [class_names[i] for i in order]

    fig_w = max(12, len(sorted_names) * 0.5)

    fig, ax = plt.subplots(figsize=(fig_w, 5), dpi=130)

    colors = [
        "#EF5350" if a < 0.5 else
        "#FFA726" if a < 0.8 else
        "#66BB6A"
        for a in sorted_acc
    ]

    ax.bar(range(len(sorted_acc)), sorted_acc, color=colors)
    ax.axhline(0.85, linestyle="--", color="blue", label="85% target")

    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class Accuracy (VGGFace2)")
    ax.set_ylabel("Accuracy")

    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=6)

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    out = os.path.join(config.RESULTS_DIR, "per_class_accuracy.png")
    plt.savefig(out)
    plt.close()

    logger.info(f"Saved per-class plot → {out}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from dataset import load_all

    train_ds, val_ds, test_ds, y_test, le, num_classes, _ = load_all()

    if not os.path.exists(config.CHECKPOINT_PATH):
        raise FileNotFoundError("Train model first.")

    model = tf.keras.models.load_model(config.CHECKPOINT_PATH)

    evaluate(model, test_ds, y_test, le)

    logger.info("Evaluation complete.")