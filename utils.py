"""
VisiGuard Utilities
===================
Shared helpers: directory creation, logging, plot saving.
Keeps every other module clean and import-light.
"""

import os
import logging
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Headless rendering – safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix

import config


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str = "visiguard") -> logging.Logger:
    """Return a logger that writes to stdout with a clean format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# Setup global module logger securely
logger = get_logger()


# ─────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all required project directories if they don't exist."""
    for path in [config.DATA_DIR, config.MODEL_DIR, config.RESULTS_DIR]:
        os.makedirs(path, exist_ok=True)
    logger.info("Project directories verified.")


# ─────────────────────────────────────────────
# Serialization helpers
# ─────────────────────────────────────────────

def save_pickle(obj, path: str) -> None:
    """Persist any Python object with pickle safely using highest protocol."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved asset entry → {path}")


def load_pickle(path: str):
    """Load a pickled Python object."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# Training curve plots
# ─────────────────────────────────────────────

def plot_training_curves(history_phase1: dict,
                         history_phase2: dict | None = None) -> None:
    """
    Plot accuracy and loss curves for one or two training phases.
    Saves the figure to RESULTS_DIR.
    """
    # Merge phases if both provided
    if history_phase2:
        acc = history_phase1["accuracy"] + history_phase2["accuracy"]
        val_acc = history_phase1["val_accuracy"] + history_phase2["val_accuracy"]
        loss = history_phase1["loss"] + history_phase2["loss"]
        val_loss = history_phase1["val_loss"] + history_phase2["val_loss"]
        phase_boundary = len(history_phase1["accuracy"])
    else:
        acc = history_phase1["accuracy"]
        val_acc = history_phase1["val_accuracy"]
        loss = history_phase1["loss"]
        val_loss = history_phase1["val_loss"]
        phase_boundary = None

    epochs = range(1, len(acc) + 1)

    # Use a fallback size if configuration field is modified
    figsize = getattr(config, "CURVE_FIGSIZE", (12, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=150)

    # ── Accuracy ──
    ax1.plot(epochs, acc, label="Train Acc", color="#2196F3", linewidth=2)
    ax1.plot(epochs, val_acc, label="Val Acc", color="#FF5722", linewidth=2, linestyle="--")
    if phase_boundary:
        ax1.axvline(phase_boundary, color="gray", linestyle=":", linewidth=1.2, label="Fine-tune start")
    ax1.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Loss ──
    ax2.plot(epochs, loss, label="Train Loss", color="#2196F3", linewidth=2)
    ax2.plot(epochs, val_loss, label="Val Loss", color="#FF5722", linewidth=2, linestyle="--")
    if phase_boundary:
        ax2.axvline(phase_boundary, color="gray", linestyle=":", linewidth=1.2, label="Fine-tune start")
    ax2.set_title("Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("VisiGuard – Training Curves", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(config.RESULTS_DIR, "training_curves.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curves saved → {out}")


# ─────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────

# 🔥 FIX: Match signature expected by evaluate.py (accepting true/predicted arrays directly)
def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    """
    Generate, normalize, and save a high-resolution confusion matrix heatmap.
    Dynamically scales bounds to prevent overlapping text labels on massive identity sets.
    """
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize row-wise (Recall fraction) safely to avoid dividing by zero
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    # 🔥 FIX: Dynamically adapt layout proportions based on class density variations
    fig_w = max(12, n * 0.25)
    fig_h = max(10, n * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Clean and shorten long directory folder name strings
    short_names = [str(name).split("_")[0] for name in class_names]

    # Adjust tick density based on class volume size parameters
    if n <= 100:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, rotation=90, ha="right", fontsize=max(4, 10 - n // 10))
        ax.set_yticklabels(short_names, fontsize=max(4, 10 - n // 10))
    else:
        ax.set_xlabel("Predicted Identities (Dense Layout Mode)", fontsize=10)
        ax.set_ylabel("True Identities (Dense Layout Mode)", fontsize=10)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    # 🔥 FIX: Draw cell values only if it fits safely within visual threshold blocks
    if n <= 30:
        thresh = cm_norm.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = cm_norm[i, j]
                if val > 0.005:  # Skip drawing zero cells to keep heatmap clean
                    ax.text(j, i, f"{val:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if val > thresh else "black")

    ax.set_title("Confusion Matrix (Row-Normalized)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    out = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved successfully → {out}")


# ─────────────────────────────────────────────
# Class distribution bar chart
# ─────────────────────────────────────────────

def plot_class_distribution(labels: list[str],
                            title: str = "Class Distribution") -> None:
    """Bar chart of sample counts per identity (top 40 shown)."""
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)          # Sort descending
    unique, counts = unique[order], counts[order]

    # Cap at 40 classes for visualization clarity
    if len(unique) > 40:
        unique = unique[:40]
        counts = counts[:40]
        title += " (Top 40 Classes Visualized)"

    fig, ax = plt.subplots(figsize=(14, 5), dpi=130)
    ax.bar(range(len(unique)), counts, color="#2196F3", edgecolor="white", width=0.7)
    
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([str(u).replace("_", " ") for u in unique],
                       rotation=45, ha="right", fontsize=8)
    
    ax.set_ylabel("Image Count Profile")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    out = os.path.join(config.RESULTS_DIR, "class_distribution.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Class distribution chart saved successfully → {out}")