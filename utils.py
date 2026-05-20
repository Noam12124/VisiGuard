"""
utils.py — Shared utility functions.

Covers:
  • Mixed precision setup
  • Reproducibility seed
  • Directory creation
  • Image loading for inference
  • Cosine similarity
  • Gallery embedding builder
  • Training curve plotting
  • Embedding visualisation (t-SNE / UMAP)
"""

import os
import json
import random
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


# ── Reproducibility ───────────────────────────────────────────────────────

def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Directory setup ───────────────────────────────────────────────────────

def ensure_dirs():
    for d in [
        config.CHECKPOINT_DIR,
        config.LOG_DIR,
        config.OUTPUT_DIR,
        config.DATA_DIR,
        config.GALLERY_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


# ── Mixed precision ───────────────────────────────────────────────────────

def setup_mixed_precision() -> bool:
    """
    Enable float16 mixed precision if the GPU supports it and
    config.MIXED_PRECISION is True.

    Returns True if mixed precision was enabled.
    """
    if not config.MIXED_PRECISION:
        print("[utils] Mixed precision disabled (config.MIXED_PRECISION=False).")
        return False

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[utils] No GPU found; mixed precision disabled.")
        return False

    # Check compute capability via nvidia-smi or TF device details
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        cap = details.get("compute_capability", (0, 0))
        if cap[0] < 7:
            print(f"[utils] GPU compute capability {cap} < 7.0; "
                  "mixed precision disabled (no Tensor Cores).")
            return False
    except Exception:
        pass   # Can't determine — assume capable and proceed

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("[utils] Mixed precision (float16) ENABLED.")
    return True


# ── Image loading ─────────────────────────────────────────────────────────

def load_image_for_inference(path: str) -> np.ndarray:
    """
    Load an image from disk and resize to (H, W, 3) float32 [0, 255].
    Does NOT detect or align — use for pre-cropped face images.

    Returns (112, 112, 3) float32 array (RGB).
    """
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]),
                         interpolation=cv2.INTER_CUBIC)
    return img_rgb.astype(np.float32)


def bgr_to_model_input(face_crop_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR face crop to a float32 RGB image in [0, 255].
    """
    rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]),
                     interpolation=cv2.INTER_CUBIC)
    return rgb.astype(np.float32)


# ── Cosine similarity ─────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D embeddings.
    Both are assumed to be L2-normalised (dot product = cosine sim).
    """
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def pairwise_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities between rows of A and B.

    Args:
        A: (N, D) L2-normalised embeddings.
        B: (M, D) L2-normalised embeddings.

    Returns:
        (N, M) similarity matrix.
    """
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T


# ── Gallery builder ───────────────────────────────────────────────────────

def build_gallery_embeddings(
    gallery_dir: str,
    model,
    detector,
    batch_size: int = 32,
) -> tuple[list[str], list[np.ndarray]]:
    """
    Build a gallery of (name, mean_embedding) pairs from a directory.

    Gallery layout:
        gallery_dir/
            Alice_Smith/   ← one or more face images
            Bob_Jones/
            ...

    For each identity, all images are embedded and the mean embedding
    (re-normalised) is used as the representative vector.  Mean pooling
    is more robust to single-image outliers than using one image.

    Returns:
        names:      list[str]          — identity names (sorted)
        embeddings: list[np.ndarray]   — one (512,) embedding per identity
    """
    from inference import embed_face_crop

    if not os.path.exists(gallery_dir):
        print(f"[gallery] Gallery directory not found: {gallery_dir}")
        return [], []

    identities = sorted([
        d for d in os.listdir(gallery_dir)
        if os.path.isdir(os.path.join(gallery_dir, d))
    ])

    names      = []
    embeddings = []

    for identity in identities:
        folder = os.path.join(gallery_dir, identity)
        image_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        if not image_files:
            continue

        identity_embs = []
        for img_path in image_files:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            # Detect largest face; if none, use full image as crop
            det = detector.detect_largest(img_bgr)
            if det is not None:
                crop = det["face_crop"]
            else:
                # Fallback: resize whole image
                from aligner import FaceAligner
                crop = FaceAligner().crop_and_resize(img_bgr, (0, 0, img_bgr.shape[1], img_bgr.shape[0]))

            emb = embed_face_crop(crop, model)
            identity_embs.append(emb)

        if not identity_embs:
            continue

        # Mean embedding (re-normalised)
        mean_emb = np.mean(identity_embs, axis=0)
        norm     = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        names.append(identity)
        embeddings.append(mean_emb.astype(np.float32))

    return names, embeddings


# ── Class catalogue helper ────────────────────────────────────────────────

def load_class_catalogue_safe(path: str = None) -> list[str]:
    """Load class catalogue JSON; return empty list if not found."""
    if path is None:
        path = os.path.join(config.CHECKPOINT_DIR, "class_names.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


# ── Training curve plotting ───────────────────────────────────────────────

def plot_training_history(
    histories: dict,
    save_dir:  str = config.OUTPUT_DIR,
):
    """
    Plot loss + accuracy curves for one or two training phases.

    Args:
        histories: dict with keys "phase1" and/or "phase2",
                   each mapping to a Keras history.history dict.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Concatenate phases for a continuous x-axis
    all_loss     = []
    all_val_loss = []
    all_acc      = []
    all_val_acc  = []
    phase_splits = [0]

    for phase_key in ["phase1", "phase2"]:
        if phase_key not in histories:
            continue
        h = histories[phase_key]
        all_loss.extend(h.get("loss", []))
        all_val_loss.extend(h.get("val_loss", []))
        all_acc.extend(h.get("accuracy", []))
        all_val_acc.extend(h.get("val_accuracy", []))
        phase_splits.append(len(all_loss))

    epochs = range(1, len(all_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    ax1.plot(epochs, all_loss,     label="Train loss",      lw=2)
    ax1.plot(epochs, all_val_loss, label="Val loss",        lw=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("ArcFace Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, all_acc,     label="Train accuracy",  lw=2)
    ax2.plot(epochs, all_val_acc, label="Val accuracy",    lw=2, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Classification Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Mark phase boundary
    if len(phase_splits) == 3:
        split_epoch = phase_splits[1]
        for ax in [ax1, ax2]:
            ax.axvline(split_epoch + 0.5, color="gray", linestyle=":", lw=1.5,
                       label="Phase 2 starts")

    fig.suptitle("Training History", fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(save_dir, "training_history.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[utils] Training history saved to {out_path}")


# ── Embedding visualisation ───────────────────────────────────────────────

def visualise_embeddings(
    embeddings:  np.ndarray,
    labels:      list[int],
    class_names: list[str] = None,
    method:      str       = "tsne",     # "tsne" or "umap"
    max_points:  int       = 3000,
    save_path:   str       = None,
):
    """
    Visualise high-dimensional embeddings in 2D (t-SNE or UMAP).

    Args:
        embeddings:  (N, 512) array.
        labels:      (N,) integer class labels.
        class_names: List of class name strings (for legend).
        method:      "tsne" or "umap".
        max_points:  Subsample to this many points for speed.
        save_path:   If given, save the figure here.
    """
    if len(embeddings) > max_points:
        idx = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[idx]
        labels     = [labels[i] for i in idx]

    labels_arr = np.array(labels)

    # Reduce to 2D
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=config.RANDOM_SEED)
            coords  = reducer.fit_transform(embeddings)
        except ImportError:
            print("[vis] UMAP not installed; falling back to t-SNE. "
                  "Install with: pip install umap-learn")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=config.RANDOM_SEED)
        coords  = reducer.fit_transform(embeddings)

    # Plot
    unique_labels = sorted(set(labels_arr.tolist()))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, lbl in enumerate(unique_labels):
        mask = labels_arr == lbl
        name = class_names[lbl] if class_names and lbl < len(class_names) else str(lbl)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], label=name, s=10, alpha=0.7)

    ax.set_title(f"Embedding Space ({method.upper()})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    if len(unique_labels) <= 20:
        ax.legend(fontsize=7, markerscale=2, loc="best")
    ax.grid(True, alpha=0.2)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[vis] Embedding visualisation saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
