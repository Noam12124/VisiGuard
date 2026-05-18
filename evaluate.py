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


def evaluate(model, test_ds, y_test, le):

    utils.ensure_dirs()
    class_names = list(le.classes_)

    logger.info("Evaluating model …")

    # ─────────────────────────────────────────────────────────────
    # 1. MANUAL LOSS + ACCURACY + PREDICTIONS USING ARCFACE WEIGHTS
    # ─────────────────────────────────────────────────────────────
    w_path = os.path.join(config.MODEL_DIR, "arcface_weights.npy")
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"ArcFace weights file not found at {w_path}. Cannot evaluate.")

    # Load and normalize ArcFace weight matrix
    W_matrix = np.load(w_path)
    W_tensor = tf.convert_to_tensor(W_matrix, dtype=tf.float32)
    W_norm = tf.nn.l2_normalize(W_tensor, axis=0)

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_trues = []

    logger.info("Generating predictions and calculating metrics over test set …")
    
    # Iterate through the test dataset batches manually to avoid compilation requirement
    for images, labels in test_ds:
        embeddings = model(images, training=False)
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)

        # Compute cosine similarity and scale to get logits
        cosine = tf.matmul(embeddings, W_norm)
        logits = cosine * 64.0  # ArcFace Scale factor standard

        # Compute Categorical Crossentropy Loss from logits
        loss_batch = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )

        total_loss += tf.reduce_sum(loss_batch).numpy()
        total_samples += tf.shape(images)[0].numpy()

        preds_batch = tf.argmax(logits, axis=1).numpy()
        all_preds.extend(preds_batch)
        all_trues.extend(labels.numpy())

    loss = total_loss / total_samples
    y_pred = np.array(all_preds)
    y_true = np.array(all_trues)
    acc = np.mean(y_pred == y_true)

    logger.info(f"Test loss: {loss:.4f}")
    logger.info(f"Test accuracy: {acc * 100:.2f}%")

    # safety alignment check
    min_len = min(len(y_pred), len(y_true))
    if len(y_pred) != len(y_true):
        logger.warning(f"Mismatch → pred={len(y_pred)} true={len(y_true)} → trimming")
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

    # ─────────────────────────────
    # 3. METRICS REPORT
    # ─────────────────────────────
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "loss": loss,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

    logger.info(
        f"\n────────────────────────\n"
        f"Evaluation Summary\n"
        f"────────────────────────\n"
        f"Loss      : {loss:.4f}\n"
        f"Accuracy  : {acc*100:.2f}%\n"
        f"Precision : {precision*100:.2f}%\n"
        f"Recall    : {recall*100:.2f}%\n"
        f"F1        : {f1*100:.2f}%\n"
        f"────────────────────────"
    )

    _plot_per_class_accuracy(y_true, y_pred, class_names)

    return metrics


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

    fig_w = max(12, len(sorted_names) * 0.4)

    fig, ax = plt.subplots(figsize=(fig_w, 5), dpi=130)

    colors = [
        "#EF5350" if a < 0.5 else
        "#FFA726" if a < 0.8 else
        "#66BB6A"
        for a in sorted_acc
    ]

    ax.bar(range(len(sorted_acc)), sorted_acc, color=colors)
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.6, label="Target (85%)")
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels([n.replace("_", " ") for n in sorted_names], rotation=90, fontsize=6)
    ax.set_title("Per-Class Test Accuracy (Sorted)", fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    out_path = os.path.join(config.RESULTS_DIR, "per_class_accuracy.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Per-class accuracy plot saved → {out_path}")