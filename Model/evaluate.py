"""
evaluate.py — Full verification evaluation: ROC, AUC, TAR@FAR, EER.

Usage:
    python evaluate.py
    python evaluate.py --model checkpoints/best_embedding_model.keras
    python evaluate.py --data-dir data/faces
    python evaluate.py --pairs-csv my_pairs.csv   # custom pair list

Why verification metrics matter:
  Classification accuracy on training identities is a proxy measure.
  Real-world face recognition is a VERIFICATION task:
    "Are these two faces the same person?"
  We evaluate this by computing cosine similarities for thousands of
  genuine (same-person) and impostor (different-person) pairs, then
  measuring:

    • AUC   — overall discriminability (higher = better)
    • EER   — threshold where FAR == FRR (lower = better)
    • TAR @ FAR=0.1%  — recall at very tight precision (higher = better)
    • TAR @ FAR=1%    — recall at standard operating point

Typical targets on LFW:
    AUC  ≥ 0.99,  EER  ≤ 2%,  TAR@FAR=1% ≥ 98%
"""

import os
import argparse
import json
import csv

import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt

import config
from dataset  import build_verification_pairs
from utils    import cosine_similarity, load_image_for_inference


# ── Embedding extraction ───────────────────────────────────────────────────

def extract_embeddings_batch(
    image_paths: list[str],
    model,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract L2-normalised embeddings for a list of image paths.

    Returns:
        embeddings: (N, 512) float32 numpy array.
    """
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = np.stack([
            load_image_for_inference(p)
            for p in batch_paths
        ], axis=0)
        embs = model.predict(imgs, verbose=0)
        all_embs.append(embs)
        if (i // batch_size) % 10 == 0:
            print(f"  [eval] Processed {i + len(batch_paths):,} / {len(image_paths):,}", end="\r")

    print()
    return np.concatenate(all_embs, axis=0).astype(np.float32)


# ── Metric computation ────────────────────────────────────────────────────

def compute_verification_metrics(
    similarities: np.ndarray,
    labels:       np.ndarray,
    n_thresholds: int = config.EER_THRESHOLD_STEPS,
    tar_far_targets: list[float] = [0.001, 0.01, 0.1],
    save_dir:     str = config.OUTPUT_DIR,
    tag:          str = "",
) -> dict:
    """
    Compute full verification metrics from similarity scores + binary labels.

    Args:
        similarities: (N,) cosine similarity scores.
        labels:       (N,) binary labels (1=same, 0=different).
        n_thresholds: Number of threshold steps for EER search.
        tar_far_targets: FAR operating points to report TAR at.
        save_dir:     Where to save the ROC plot.
        tag:          Optional string suffix for plot filenames.

    Returns:
        dict with keys: auc, eer, eer_threshold, tar_at_far (dict).
    """
    os.makedirs(save_dir, exist_ok=True)

    labels      = np.array(labels,      dtype=int)
    similarities = np.array(similarities, dtype=np.float32)

    # ── ROC curve ──────────────────────────────────────────────────────────
    # FPR: False Positive Rate, TPR: True Positive Rate
    fpr, tpr, thresholds = roc_curve(labels, similarities, pos_label=1)
    roc_auc              = auc(fpr, tpr)

    # ── EER ────────────────────────────────────────────────────────────────
    fnr  = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    eer_idx       = np.argmin(diff)
    eer           = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx]) if eer_idx < len(thresholds) else 0.5

    # ── TAR @ FAR ──────────────────────────────────────────────────────────
    tar_at_far = {}
    for target_far in tar_far_targets:
        # Find the highest TAR where FAR ≤ target
        mask = fpr <= target_far
        if mask.any():
            tar = float(tpr[mask][-1])
        else:
            tar = 0.0
        tar_at_far[f"TAR@FAR={target_far}"] = tar

    metrics = {
        "auc":           float(roc_auc),
        "eer":           eer,
        "eer_threshold": eer_threshold,
        "tar_at_far":    tar_at_far,
    }

    # ── Print ──────────────────────────────────────────────────────────────
    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  Verification Metrics{' (' + tag + ')' if tag else ''}")
    print(sep)
    print(f"  AUC:           {roc_auc:.4f}")
    print(f"  EER:           {eer * 100:.2f}%  (threshold = {eer_threshold:.4f})")
    for k, v in tar_at_far.items():
        print(f"  {k}:  {v * 100:.2f}%")
    print(sep)

    # ── ROC plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr * 100, tpr * 100, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.scatter(
        [eer * 100], [(1 - eer) * 100],
        s=80, color="red", zorder=5, label=f"EER = {eer*100:.2f}%"
    )
    for target_far in tar_far_targets:
        tar_val = tar_at_far[f"TAR@FAR={target_far}"]
        ax.scatter(
            [target_far * 100], [tar_val * 100],
            s=60, marker="x", zorder=5,
            label=f"TAR={tar_val*100:.1f}% @ FAR={target_far*100:.1f}%",
        )
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("TAR (%)")
    ax.set_title("Face Verification ROC Curve")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    fname = f"roc_curve{'_' + tag if tag else ''}.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] ROC plot saved to {os.path.join(save_dir, fname)}")

    # ── Similarity distribution plot ───────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    genuine  = similarities[labels == 1]
    impostor = similarities[labels == 0]
    ax2.hist(genuine,  bins=80, alpha=0.6, color="#2ecc71", density=True,
             label=f"Genuine  (n={len(genuine):,})")
    ax2.hist(impostor, bins=80, alpha=0.6, color="#e74c3c", density=True,
             label=f"Impostor (n={len(impostor):,})")
    ax2.axvline(eer_threshold, color="navy", lw=2, linestyle="--",
                label=f"EER threshold = {eer_threshold:.3f}")
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Density")
    ax2.set_title("Similarity Score Distributions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fname2 = f"similarity_dist{'_' + tag if tag else ''}.png"
    fig2.savefig(os.path.join(save_dir, fname2), dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"[eval] Distribution plot saved to {os.path.join(save_dir, fname2)}")

    return metrics


# ── Precision-Recall + decision threshold analysis ────────────────────────

def threshold_sweep(
    similarities: np.ndarray,
    labels:       np.ndarray,
    n_steps:      int = 200,
    save_dir:     str = config.OUTPUT_DIR,
) -> dict:
    """
    Sweep decision thresholds and find the F1-optimal one.

    Returns dict with keys: optimal_threshold, precision, recall, f1.
    """
    from sklearn.metrics import precision_recall_fscore_support

    thresholds  = np.linspace(similarities.min(), similarities.max(), n_steps)
    best_f1     = 0.0
    best_thresh = 0.5
    best_pr = best_re = 0.0

    for t in thresholds:
        preds = (similarities >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        if f > best_f1:
            best_f1, best_thresh, best_pr, best_re = f, t, p, r

    result = {
        "optimal_threshold": float(best_thresh),
        "precision":         float(best_pr),
        "recall":            float(best_re),
        "f1":                float(best_f1),
    }
    print(f"[eval] Optimal threshold: {best_thresh:.4f}  "
          f"P={best_pr*100:.1f}%  R={best_re*100:.1f}%  F1={best_f1*100:.1f}%")
    return result


# ── Helper function for Lambda layer ───────────────────────────────────────

def _l2_norm(x):
    """L2 normalization helper used by the embedding layer."""
    return tf.nn.l2_normalize(x, axis=-1)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate face verification performance.")
    p.add_argument("--model",    default=config.BEST_EMBEDDING_MODEL,
                   help="Path to the saved embedding model (.keras).")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--pairs-csv", default=None,
                   help="Optional CSV with columns: path1,path2,label")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--tag",      default="", help="Tag for output filenames.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load model ─────────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[eval] Model not found at {args.model}. "
              "Run train.py first.")
        return

    print(f"[eval] Loading model from {args.model}…")
    
    # הוספנו את פונקציית הלמבדה לחפצים המותאמים אישית (custom_objects)
    model = tf.keras.models.load_model(
        args.model,
        compile=False,
        custom_objects={
            "ArcFaceLayer": __import__("arcface").ArcFaceLayer,
            "_l2_norm": _l2_norm
        },
    )
    print("[eval] Model loaded.")

    # ── Build verification pairs ───────────────────────────────────────────
    if args.pairs_csv and os.path.exists(args.pairs_csv):
        print(f"[eval] Loading pairs from {args.pairs_csv}…")
        paths1, paths2, pair_labels = [], [], []
        with open(args.pairs_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                paths1.append(row["path1"])
                paths2.append(row["path2"])
                pair_labels.append(int(row["label"]))
    else:
        data_dir = args.data_dir
        if data_dir is None:
            aligned = config.DATA_DIR.rstrip("/\\") + "_aligned"
            data_dir = aligned if os.path.exists(aligned) else config.DATA_DIR
        paths1, paths2, pair_labels = build_verification_pairs(data_dir=data_dir)

    print(f"[eval] Computing embeddings for {len(paths1) * 2:,} images…")

    # Extract embeddings for all unique images (avoid re-computing duplicates)
    all_paths  = paths1 + paths2
    unique_paths = list(dict.fromkeys(all_paths))   # preserve order, deduplicate
    path_to_idx  = {p: i for i, p in enumerate(unique_paths)}

    all_embs = extract_embeddings_batch(unique_paths, model, batch_size=args.batch_size)

    # Build similarity array
    embs1 = all_embs[[path_to_idx[p] for p in paths1]]
    embs2 = all_embs[[path_to_idx[p] for p in paths2]]

    # Cosine similarity = dot product of L2-normalised vectors
    sims = np.sum(embs1 * embs2, axis=1)   # (N,)

    # ── Compute metrics ────────────────────────────────────────────────────
    metrics = compute_verification_metrics(
        similarities = sims,
        labels       = pair_labels,
        save_dir     = config.OUTPUT_DIR,
        tag          = args.tag,
    )

    thresh_metrics = threshold_sweep(sims, np.array(pair_labels))
    metrics.update(thresh_metrics)

    # ── Save JSON ──────────────────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR,
                            f"eval_metrics{'_' + args.tag if args.tag else ''}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Metrics saved to {out_path}")


if __name__ == "__main__":
    main()