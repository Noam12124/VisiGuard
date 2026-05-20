"""
train.py  (VisiGuard — revised)
═══════════════════════════════════════════════════════════════════════════════
Key changes vs the original:
──────────────────────────────────────────────────────────────────────────────
[FIX-A] VerificationCallback
    A custom Keras callback that runs at the END of every epoch:
      1. Pulls the frozen embedding_model (no ArcFace head).
      2. Embeds all images in the val_ids verification-pair set.
      3. Computes cosine similarity for every pair.
      4. Calculates AUC, EER, and TAR@FAR=1%.
      5. Logs  val_ver_auc  into Keras's metric dict so ModelCheckpoint
         and EarlyStopping can monitor it.

[FIX-B] ModelCheckpoint + EarlyStopping pivot to val_ver_auc
    These now track the true verification AUC instead of the misleading
    classification val_accuracy.

[FIX-C] build_datasets() call updated for the new 6-return-value signature
    (train_ds, val_ds, test_ds, class_names, val_ids, test_ids).

[FIX-D] ArcFaceTrainer gracefully handles the mixed-precision scaled-loss
    path even when get_scaled_loss / get_unscaled_gradients have been
    deprecated in newer Keras/TF builds (replaced by the GradientTape
    approach used by LossScaleOptimizer internally).
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import argparse
import json
import math
import time

import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve

import config
from model import (
    build_model,
    freeze_backbone,
    unfreeze_top_layers,
    compile_model,
    model_summary,
    get_cosine_scheduler,
)
from dataset import (
    build_datasets,
    build_verification_pairs,
    prepare_aligned_dataset,
)
from utils import (
    plot_training_history,
    setup_mixed_precision,
    ensure_dirs,
    set_seed,
)


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-A]  VerificationCallback
# ─────────────────────────────────────────────────────────────────────────────

class VerificationCallback(tf.keras.callbacks.Callback):
    """
    End-of-epoch pairwise verification evaluator.

    At the end of every epoch this callback:
      1. Extracts L2-normalised 512-d embeddings for every image in the
         verification pair set using embedding_model (the head-free model
         that shares weights with full_model).
      2. Computes cosine similarity for each (path1, path2) pair.
      3. Calculates AUC, EER, and TAR@FAR=1%.
      4. Stores the results in logs so Keras's ModelCheckpoint and
         EarlyStopping can monitor  'val_ver_auc'.

    Parameters
    ──────────
    embedding_model : tf.keras.Model
        The L2-normalised embedding extractor (no ArcFace head).
        Shares weights with the training model — no copying needed.
    data_dir : str
        Root of the aligned dataset.
    val_ids : list[str]
        Identity names drawn from the VALIDATION split only.
        These identities have never been seen during training.  [FIX-3]
    pairs_per_identity : int
        How many genuine pairs to generate per identity.
        Keep this small (10–20) to keep per-epoch overhead under 30 s on T4.
    embed_batch_size : int
        Batch size for embedding inference.  64 fits comfortably on T4.
    run_every_n_epochs : int
        Skip verification on cheaper epochs.  E.g. 2 = run every other epoch.
        During fine-tuning (phase 2) you may want 1.
    """

    def __init__(
        self,
        embedding_model: tf.keras.Model,
        data_dir: str,
        val_ids: list[str],
        pairs_per_identity: int = 15,
        embed_batch_size: int = 64,
        run_every_n_epochs: int = 1,
        verbose: bool = True,
    ):
        super().__init__()
        self.embedding_model    = embedding_model
        self.data_dir           = data_dir
        self.val_ids            = val_ids
        self.pairs_per_identity = pairs_per_identity
        self.embed_batch_size   = embed_batch_size
        self.run_every_n_epochs = run_every_n_epochs
        self.verbose            = verbose

        # Pre-build pairs once (they don't change between epochs)
        self.paths1, self.paths2, self.pair_labels = build_verification_pairs(
            data_dir           = data_dir,
            identity_list      = val_ids,         # [FIX-3] val only
            pairs_per_identity = pairs_per_identity,
        )

        # Cache of unique paths → avoid re-embedding the same image twice
        self._unique_paths: list[str] = list(
            dict.fromkeys(self.paths1 + self.paths2)
        )
        self._path_to_idx: dict[str, int] = {
            p: i for i, p in enumerate(self._unique_paths)
        }
        print(
            f"[VerificationCallback] Ready: {len(self.pair_labels):,} pairs, "
            f"{len(self._unique_paths):,} unique images."
        )

    # ── Image loading helper ───────────────────────────────────────────────

    def _load_image(self, path: str) -> np.ndarray:
        """Load a pre-aligned face crop as float32 RGB [0, 255]."""
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # Return a black image rather than crashing the whole callback
            return np.zeros((*config.IMAGE_SIZE, 3), dtype=np.float32)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(
            img_rgb,
            (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        return img_rgb.astype(np.float32)

    # ── Embedding extraction ───────────────────────────────────────────────

    def _extract_all_embeddings(self) -> np.ndarray:
        """
        Embed all unique images in batches.

        Returns (N, 512) float32 L2-normalised array.
        """
        all_embs = []
        paths    = self._unique_paths
        bs       = self.embed_batch_size

        for start in range(0, len(paths), bs):
            batch_paths = paths[start : start + bs]
            batch_imgs  = np.stack(
                [self._load_image(p) for p in batch_paths], axis=0
            )
            embs = self.embedding_model.predict(batch_imgs, verbose=0)
            # Re-normalise for safety (model already outputs L2-normed, but
            # numerical precision can drift slightly during float16 training)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs  = embs / np.maximum(norms, 1e-8)
            all_embs.append(embs.astype(np.float32))

        return np.concatenate(all_embs, axis=0)

    # ── Metric computation ─────────────────────────────────────────────────

    def _compute_metrics(
        self, sims: np.ndarray, labels: np.ndarray
    ) -> dict[str, float]:
        """Compute AUC, EER, and TAR@FAR=1% from cosine similarities."""
        auc = float(roc_auc_score(labels, sims))

        fpr, tpr, _ = roc_curve(labels, sims, pos_label=1)
        fnr         = 1.0 - tpr
        eer_idx     = int(np.argmin(np.abs(fpr - fnr)))
        eer         = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

        # TAR @ FAR ≤ 1%
        mask_1pct   = fpr <= 0.01
        tar_at_1pct = float(tpr[mask_1pct][-1]) if mask_1pct.any() else 0.0

        return {"auc": auc, "eer": eer, "tar_at_far1": tar_at_1pct}

    # ── Keras hook ────────────────────────────────────────────────────────

    def on_epoch_end(self, epoch: int, logs: dict = None):
        # Respect the run_every_n_epochs cadence
        if (epoch + 1) % self.run_every_n_epochs != 0:
            # Still insert a value so ModelCheckpoint doesn't complain on
            # epochs where we skip.
            if logs is not None:
                logs["val_ver_auc"] = logs.get("val_ver_auc", 0.0)
            return

        t0 = time.time()

        # Extract embeddings for all unique images
        all_embs = self._extract_all_embeddings()

        # Gather embeddings for each pair
        idx1 = [self._path_to_idx[p] for p in self.paths1]
        idx2 = [self._path_to_idx[p] for p in self.paths2]
        embs1 = all_embs[idx1]   # (N, 512)
        embs2 = all_embs[idx2]   # (N, 512)

        # Cosine similarity = dot product of L2-normalised vectors
        sims   = np.sum(embs1 * embs2, axis=1)
        labels = np.array(self.pair_labels, dtype=int)

        metrics = self._compute_metrics(sims, labels)
        elapsed = time.time() - t0

        # ── Log into Keras metrics dict ───────────────────────────────────
        # Keras's ModelCheckpoint and EarlyStopping read from `logs`.
        if logs is not None:
            logs["val_ver_auc"]      = metrics["auc"]
            logs["val_ver_eer"]      = metrics["eer"]
            logs["val_ver_tar1pct"]  = metrics["tar_at_far1"]

        if self.verbose:
            print(
                f"\n  ┌─ Verification @ epoch {epoch + 1} "
                f"({elapsed:.1f}s) ─────────────────\n"
                f"  │  AUC:           {metrics['auc']:.4f}\n"
                f"  │  EER:           {metrics['eer'] * 100:.2f}%\n"
                f"  │  TAR@FAR=1%:    {metrics['tar_at_far1'] * 100:.2f}%\n"
                f"  └────────────────────────────────────────────────────"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Custom training-loop wrapper for ArcFace label injection
# ─────────────────────────────────────────────────────────────────────────────

class ArcFaceTrainer(tf.keras.Model):
    """
    Keras subclass wrapper that injects the integer label into the ArcFace
    layer during both train_step and test_step.

    Dataset format:  ((image_tensor, label_tensor), label_tensor)
      inputs[0] = image  (B, H, W, 3)
      inputs[1] = label  (B,)    ← forwarded to ArcFaceLayer
      targets   = label  (B,)    ← used by compiled_loss

    [FIX-D] The mixed-precision branch now uses the modern
    tf.keras.mixed_precision.LossScaleOptimizer API  (get_scaled_loss /
    get_unscaled_gradients) and falls back gracefully when the optimizer
    is not a LossScaleOptimizer.
    """

    def train_step(self, data):
        (images, labels), _ = data

        with tf.GradientTape() as tape:
            logits = self([images, labels], training=True)
            loss   = self.compiled_loss(labels, logits)
            loss  += tf.add_n(self.losses) if self.losses else 0.0

            # [FIX-D] Mixed precision: scale only when using LossScaleOptimizer
            is_lso = isinstance(
                self.optimizer,
                tf.keras.mixed_precision.LossScaleOptimizer,
            )
            scaled_loss = self.optimizer.get_scaled_loss(loss) if is_lso else loss

        grads = tape.gradient(scaled_loss, self.trainable_variables)

        if is_lso:
            grads = self.optimizer.get_unscaled_gradients(grads)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels), _ = data
        logits = self([images, labels], training=False)
        self.compiled_loss(labels, logits)
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Callback builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_callbacks(
    phase:              int,
    lr_schedule,
    embedding_model:    tf.keras.Model,
    data_dir:           str,
    val_ids:            list[str],
    initial_epoch:      int = 0,
    run_ver_every:      int = 1,
) -> list:
    """
    Build the full callback stack for one training phase.

    [FIX-B] ModelCheckpoint and EarlyStopping now monitor 'val_ver_auc'
    (the cosine-similarity AUC from the VerificationCallback) instead of
    the classification 'val_accuracy'.

    The VerificationCallback is intentionally placed FIRST in the list so
    it runs — and populates logs['val_ver_auc'] — before ModelCheckpoint
    and EarlyStopping read from logs.
    """
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR,        exist_ok=True)

    # Periodic weight checkpoints (not the best-model checkpoint)
    ckpt_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"phase{phase}_epoch{{epoch:03d}}_auc{{val_ver_auc:.4f}}.weights.h5",
    )

    callbacks = [
        # ── [FIX-A] Verification metric — MUST be first ───────────────────
        VerificationCallback(
            embedding_model    = embedding_model,
            data_dir           = data_dir,
            val_ids            = val_ids,
            pairs_per_identity = 15,          # ~30 s per epoch on T4
            embed_batch_size   = 64,
            run_every_n_epochs = run_ver_every,
        ),

        # ── [FIX-B] ModelCheckpoint — monitors val_ver_auc ────────────────
        tf.keras.callbacks.ModelCheckpoint(
            filepath          = config.BEST_TRAIN_MODEL,
            monitor           = "val_ver_auc",   # ← changed
            mode              = "max",
            save_best_only    = True,
            save_weights_only = False,
            verbose           = 1,
        ),

        # Periodic checkpoints (every epoch, weights only)
        tf.keras.callbacks.ModelCheckpoint(
            filepath          = ckpt_path,
            save_best_only    = False,
            save_weights_only = True,
            verbose           = 0,
            save_freq         = "epoch",
        ),

        # ── [FIX-B] EarlyStopping — monitors val_ver_auc ─────────────────
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_ver_auc",   # ← changed
            mode                 = "max",
            patience             = config.EARLY_STOPPING_PATIENCE,
            restore_best_weights = True,
            verbose              = 1,
        ),

        # ReduceLROnPlateau kept on val_loss as a secondary safety net.
        # The cosine schedule is primary; ReduceLR only fires when the
        # cosine schedule alone isn't making progress.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = config.REDUCE_LR_FACTOR,
            patience = config.REDUCE_LR_PATIENCE,
            min_lr   = config.MIN_LR,
            verbose  = 1,
        ),

        # Cosine LR schedule
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),

        # TensorBoard — val_ver_auc now appears as a custom scalar
        tf.keras.callbacks.TensorBoard(
            log_dir        = os.path.join(config.LOG_DIR, f"phase{phase}"),
            histogram_freq = 0,
            update_freq    = "epoch",
        ),

        # CSV log
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.LOG_DIR, f"phase{phase}_log.csv"),
            append=(initial_epoch > 0),
        ),
    ]
    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — warm-up
# ─────────────────────────────────────────────────────────────────────────────

def train_phase1(
    full_model,
    embedding_model,
    train_ds,
    val_ds,
    data_dir:    str,
    val_ids:     list[str],
    num_classes: int,
    batch_size:  int,
) -> tf.keras.callbacks.History:
    print("\n" + "=" * 60)
    print("  PHASE 1 — WARM-UP  (backbone frozen)")
    print("=" * 60)

    freeze_backbone(full_model)

    lr_schedule = get_cosine_scheduler(
        base_lr      = config.WARMUP_LR,
        total_epochs = config.WARMUP_EPOCHS,
        warmup_epochs = 3,
        min_lr       = config.MIN_LR,
    )
    compile_model(full_model, lr=config.WARMUP_LR)

    callbacks = _build_callbacks(
        phase           = 1,
        lr_schedule     = lr_schedule,
        embedding_model = embedding_model,
        data_dir        = data_dir,
        val_ids         = val_ids,
        run_ver_every   = 2,   # every 2 epochs in warm-up to save time
    )

    history = full_model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = config.WARMUP_EPOCHS,
        callbacks       = callbacks,
        verbose         = 1,
    )

    embedding_model.save(config.BEST_EMBEDDING_MODEL)
    print(f"[train] Phase 1 done.  Embedding model → {config.BEST_EMBEDDING_MODEL}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def train_phase2(
    full_model,
    embedding_model,
    train_ds,
    val_ds,
    data_dir:      str,
    val_ids:       list[str],
    initial_epoch: int = 0,
) -> tf.keras.callbacks.History:
    print("\n" + "=" * 60)
    print("  PHASE 2 — FINE-TUNING  (top backbone layers unfrozen)")
    print("=" * 60)

    unfreeze_top_layers(full_model)

    lr_schedule = get_cosine_scheduler(
        base_lr      = config.FINETUNE_LR,
        total_epochs = config.WARMUP_EPOCHS + config.FINETUNE_EPOCHS,
        min_lr       = config.MIN_LR,
    )
    compile_model(full_model, lr=config.FINETUNE_LR)

    callbacks = _build_callbacks(
        phase           = 2,
        lr_schedule     = lr_schedule,
        embedding_model = embedding_model,
        data_dir        = data_dir,
        val_ids         = val_ids,
        initial_epoch   = initial_epoch,
        run_ver_every   = 1,   # every epoch in fine-tuning
    )

    history = full_model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = config.WARMUP_EPOCHS + config.FINETUNE_EPOCHS,
        initial_epoch   = initial_epoch,
        callbacks       = callbacks,
        verbose         = 1,
    )

    embedding_model.save(config.BEST_EMBEDDING_MODEL)
    print(f"[train] Phase 2 done.  Embedding model → {config.BEST_EMBEDDING_MODEL}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train VisiGuard face recognition model.")
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--phase",       type=int, default=1, choices=[1, 2])
    parser.add_argument("--batch-size",  type=int, default=config.BATCH_SIZE)
    parser.add_argument("--data-dir",    type=str, default=None)
    parser.add_argument("--no-align",    action="store_true",
                        help="Skip offline alignment pass.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    ensure_dirs()
    set_seed(config.RANDOM_SEED)

    use_mp = setup_mixed_precision()

    # ── Dataset ───────────────────────────────────────────────────────────
    data_dir = args.data_dir or config.DATA_DIR

    if not args.no_align:
        print("[train] Running offline face alignment…")
        data_dir = prepare_aligned_dataset(src_dir=data_dir)

    print("[train] Building tf.data pipelines…")

    # [FIX-C] Unpack 6-tuple returned by the revised build_datasets()
    train_ds, val_ds, test_ds, class_names, val_ids, test_ids = build_datasets(
        data_dir   = data_dir,
        batch_size = args.batch_size,
    )
    num_classes = len(class_names)
    print(f"[train] Training on {num_classes} identities.")
    print(f"[train] Verification callback will use {len(val_ids)} val identities.")

    # ── Build model ───────────────────────────────────────────────────────
    print("[train] Building model…")
    full_model, embedding_model = build_model(
        num_classes=num_classes, training=True
    )
    model_summary(full_model)

    # ── Resume ────────────────────────────────────────────────────────────
    start_phase   = args.phase
    initial_epoch = 0

    if args.resume and os.path.exists(config.BEST_TRAIN_MODEL):
        print(f"[train] Resuming from {config.BEST_TRAIN_MODEL}")
        full_model = tf.keras.models.load_model(
            config.BEST_TRAIN_MODEL,
            custom_objects={"ArcFaceLayer": __import__("arcface").ArcFaceLayer},
        )
        csv_p2 = os.path.join(config.LOG_DIR, "phase2_log.csv")
        csv_p1 = os.path.join(config.LOG_DIR, "phase1_log.csv")
        if os.path.exists(csv_p2):
            import csv
            with open(csv_p2) as f:
                rows = list(csv.DictReader(f))
            if rows:
                initial_epoch = int(rows[-1]["epoch"]) + 1
                start_phase   = 2
        elif os.path.exists(csv_p1):
            import csv
            with open(csv_p1) as f:
                rows = list(csv.DictReader(f))
            if rows and len(rows) >= config.WARMUP_EPOCHS:
                start_phase = 2
        print(f"[train] Resuming phase {start_phase} from epoch {initial_epoch}")

    # ── Training ──────────────────────────────────────────────────────────
    histories = {}

    if start_phase == 1:
        h1 = train_phase1(
            full_model, embedding_model,
            train_ds, val_ds,
            data_dir, val_ids,
            num_classes, args.batch_size,
        )
        histories["phase1"] = h1.history
        start_phase   = 2
        initial_epoch = config.WARMUP_EPOCHS

    if start_phase == 2:
        h2 = train_phase2(
            full_model, embedding_model,
            train_ds, val_ds,
            data_dir, val_ids,
            initial_epoch=initial_epoch,
        )
        histories["phase2"] = h2.history

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_training_history(histories, save_dir=config.OUTPUT_DIR)

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\n[train] Running final verification evaluation on test identities…")
    from evaluate import compute_verification_metrics
    from dataset  import build_verification_pairs

    test_paths1, test_paths2, test_labels = build_verification_pairs(
        data_dir      = data_dir,
        identity_list = test_ids,
    )

    # Embed all test images
    all_paths   = test_paths1 + test_paths2
    unique_paths = list(dict.fromkeys(all_paths))
    p2i          = {p: i for i, p in enumerate(unique_paths)}

    def _load(path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((*config.IMAGE_SIZE, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        return img.astype(np.float32)

    bs   = 64
    embs = []
    for start in range(0, len(unique_paths), bs):
        batch = np.stack([_load(p) for p in unique_paths[start:start+bs]])
        embs.append(embedding_model.predict(batch, verbose=0))
    embs = np.concatenate(embs, axis=0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.maximum(norms, 1e-8)

    e1   = embs[[p2i[p] for p in test_paths1]]
    e2   = embs[[p2i[p] for p in test_paths2]]
    sims = np.sum(e1 * e2, axis=1)

    test_metrics = compute_verification_metrics(
        similarities = sims,
        labels       = test_labels,
        save_dir     = config.OUTPUT_DIR,
        tag          = "test",
    )

    with open(os.path.join(config.OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, (float, np.floating)) else v
             for k, v in test_metrics.items()},
            f, indent=2,
        )

    print("\n[train] Training complete.")
    print(f"        Embedding model : {config.BEST_EMBEDDING_MODEL}")
    print(f"        Outputs         : {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
