"""
train.py — Single-phase training pipeline for the FaceResNet model.

From-scratch training differences vs the old two-phase pretrained approach:
  • No Phase 1 (frozen backbone) / Phase 2 (unfreezing) split.
    The entire network is trainable from epoch 0.
  • Linear LR warm-up for the first WARMUP_EPOCHS to stabilise
    random-weight initialisation.  Cosine annealing for the remainder.
  • Higher initial LR (1e-2 vs 1e-3) because random weights need a
    bigger gradient step to escape the initialisation manifold.
  • More epochs (120) to compensate for not starting from ImageNet features.

Usage:
    python train.py                    # full training run
    python train.py --resume           # continue from best checkpoint
    python train.py --batch-size 32    # override batch size
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
    compile_model,
    model_summary,
    get_cosine_scheduler,
)
from dataset import (
    build_datasets,
    build_verification_pairs,
)
from utils import (
    plot_training_history,
    setup_mixed_precision,
    ensure_dirs,
    set_seed,
)


# ─────────────────────────────────────────────────────────────────────────────
# VerificationCallback — pairwise AUC/EER monitor
# ─────────────────────────────────────────────────────────────────────────────

class VerificationCallback(tf.keras.callbacks.Callback):
    """
    End-of-epoch pairwise verification evaluator.

    Computes AUC, EER, and TAR@FAR=1% on held-out validation identity pairs
    so we can track real-world verification performance, not just classification
    accuracy on training classes.
    """

    def __init__(
        self,
        embedding_model:    tf.keras.Model,
        data_dir:           str,
        val_ids:            list,
        pairs_per_identity: int = 15,
        embed_batch_size:   int = 64,
        run_every_n_epochs: int = 1,
        verbose:            bool = True,
    ):
        super().__init__()
        self.embedding_model    = embedding_model
        self.data_dir           = data_dir
        self.val_ids            = val_ids
        self.embed_batch_size   = embed_batch_size
        self.run_every_n_epochs = run_every_n_epochs
        self.verbose            = verbose

        # Pre-build pairs once
        self.paths1, self.paths2, self.pair_labels = build_verification_pairs(
            data_dir   = config.DATA_DIR,
            identities = val_ids,
            num_pairs  = config.NUM_VERIFICATION_PAIRS,
        )

        self._unique_paths = list(dict.fromkeys(self.paths1 + self.paths2))
        self._path_to_idx  = {p: i for i, p in enumerate(self._unique_paths)}

        print(
            f"[VerificationCallback] Ready: {len(self.pair_labels):,} pairs, "
            f"{len(self._unique_paths):,} unique images."
        )

    def _load_image(self, path: str) -> np.ndarray:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return np.zeros((*config.IMAGE_SIZE, 3), dtype=np.float32)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]),
                             interpolation=cv2.INTER_CUBIC)
        return img_rgb.astype(np.float32)

    def _extract_all_embeddings(self) -> np.ndarray:
        all_embs = []
        bs       = self.embed_batch_size
        for start in range(0, len(self._unique_paths), bs):
            batch = np.stack(
                [self._load_image(p) for p in self._unique_paths[start:start + bs]],
                axis=0,
            )
            embs = self.embedding_model.predict(batch, verbose=0)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            all_embs.append((embs / np.maximum(norms, 1e-8)).astype(np.float32))
        return np.concatenate(all_embs, axis=0)

    def _compute_metrics(self, sims: np.ndarray, labels: np.ndarray) -> dict:
        auc_val = float(roc_auc_score(labels, sims))
        fpr, tpr, _ = roc_curve(labels, sims, pos_label=1)
        fnr         = 1.0 - tpr
        eer_idx     = int(np.argmin(np.abs(fpr - fnr)))
        eer         = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        mask        = fpr <= 0.01
        tar_1pct    = float(tpr[mask][-1]) if mask.any() else 0.0
        return {"auc": auc_val, "eer": eer, "tar_at_far1": tar_1pct}

    def on_epoch_end(self, epoch: int, logs: dict = None):
        if (epoch + 1) % self.run_every_n_epochs != 0:
            if logs is not None:
                logs["val_ver_auc"] = logs.get("val_ver_auc", 0.0)
            return

        t0       = time.time()
        all_embs = self._extract_all_embeddings()

        embs1  = all_embs[[self._path_to_idx[p] for p in self.paths1]]
        embs2  = all_embs[[self._path_to_idx[p] for p in self.paths2]]
        sims   = np.sum(embs1 * embs2, axis=1)
        labels = np.array(self.pair_labels, dtype=int)

        metrics = self._compute_metrics(sims, labels)
        elapsed = time.time() - t0

        if logs is not None:
            logs["val_ver_auc"]     = metrics["auc"]
            logs["val_ver_eer"]     = metrics["eer"]
            logs["val_ver_tar1pct"] = metrics["tar_at_far1"]

        if self.verbose:
            print(
                f"\n  ┌─ Verification @ epoch {epoch + 1} ({elapsed:.1f}s) ───────────\n"
                f"  │  AUC:           {metrics['auc']:.4f}\n"
                f"  │  EER:           {metrics['eer'] * 100:.2f}%\n"
                f"  │  TAR@FAR=1%:    {metrics['tar_at_far1'] * 100:.2f}%\n"
                f"  └───────────────────────────────────────────────────────"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ArcFace training wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ArcFaceTrainer(tf.keras.Model):
    """
    Subclass wrapper that injects integer labels into the ArcFace layer
    during both train_step and test_step.  Keras's standard .fit() does
    not pass labels into the model call, so we override the step methods.
    """

    def train_step(self, data):
        # Unpack the data (images, labels) and the ignored target element
        (images, labels), _ = data

        with tf.GradientTape() as tape:
            # Forward pass: Compute the logits using the backbone and ArcFace layer
            logits = self([images, labels], training=True)
            
            # Keras 3 unified loss calculation (handles base loss + L2 regularization self.losses)
            loss = self.compute_loss(x=[images, labels], y=labels, y_pred=logits)
            
            # Keras 3 mixed precision scaling
            scaled_loss = self.optimizer.scale_loss(loss)

        # Compute gradients using the scaled loss
        grads = tape.gradient(scaled_loss, self.trainable_variables)

        # Apply gradients (Keras 3 automatically scales gradients down under the hood)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Explicitly update metrics to bypass Keras 3 deprecation warnings
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(labels, logits)
                
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
    embedding_model: tf.keras.Model,
    lr_schedule,
    data_dir:        str,
    val_ids:         list,
    initial_epoch:   int = 0,
) -> list:
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR,        exist_ok=True)

    ckpt_path = os.path.join(
        config.CHECKPOINT_DIR,
        "epoch{epoch:03d}_auc{val_ver_auc:.4f}.weights.h5",
    )

    return [
        VerificationCallback(
            embedding_model    = embedding_model,
            data_dir           = data_dir,
            val_ids            = val_ids,
            pairs_per_identity = 15,
            embed_batch_size   = 64,
            run_every_n_epochs = 1,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath          = config.BEST_TRAIN_MODEL,
            monitor           = "val_ver_auc",
            mode              = "max",
            save_best_only    = True,
            save_weights_only = False,
            verbose           = 1,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath          = ckpt_path,
            save_best_only    = False,
            save_weights_only = True,
            verbose           = 0,
            save_freq         = "epoch",
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_ver_auc",
            mode                 = "max",
            patience             = config.EARLY_STOPPING_PATIENCE,
            restore_best_weights = True,
            verbose              = 1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = config.REDUCE_LR_FACTOR,
            patience = config.REDUCE_LR_PATIENCE,
            min_lr   = config.MIN_LR,
            verbose  = 1,
        ),

        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),

        tf.keras.callbacks.TensorBoard(
            log_dir        = os.path.join(config.LOG_DIR, "training"),
            histogram_freq = 0,
            update_freq    = "epoch",
        ),

        tf.keras.callbacks.CSVLogger(
            os.path.join(config.LOG_DIR, "training_log.csv"),
            append=(initial_epoch > 0),
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main training function (single phase)
# ─────────────────────────────────────────────────────────────────────────────

def train(
    full_model,
    embedding_model,
    train_ds,
    val_ds,
    data_dir:      str,
    val_ids:       list,
    initial_epoch: int = 0,
) -> tf.keras.callbacks.History:
    print("\n" + "=" * 60)
    print("  TRAINING FROM SCRATCH  (full network trainable)")
    print(f"  Epochs: {initial_epoch} → {config.TOTAL_EPOCHS}")
    print(f"  LR: {config.INITIAL_LR} (warmup {config.WARMUP_EPOCHS} epochs) → {config.MIN_LR}")
    print("=" * 60)

    lr_schedule = get_cosine_scheduler(
        base_lr       = config.INITIAL_LR,
        total_epochs  = config.TOTAL_EPOCHS,
        warmup_epochs = config.WARMUP_EPOCHS,
        min_lr        = config.MIN_LR,
    )
    compile_model(full_model, lr=config.INITIAL_LR)

    callbacks = _build_callbacks(
        embedding_model = embedding_model,
        lr_schedule     = lr_schedule,
        data_dir        = data_dir,
        val_ids         = val_ids,
        initial_epoch   = initial_epoch,
    )

    history = full_model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = config.TOTAL_EPOCHS,
        initial_epoch   = initial_epoch,
        callbacks       = callbacks,
        verbose         = 1,
    )

    embedding_model.save(config.BEST_EMBEDDING_MODEL)
    print(f"[train] Done. Embedding model → {config.BEST_EMBEDDING_MODEL}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train FaceResNet face recognition model from scratch.")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from best saved checkpoint.")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                   help="Training batch size.")
    p.add_argument("--data-dir",   type=str, default=None,
                   help="Override dataset directory.")
    p.add_argument("--epochs",     type=int, default=config.TOTAL_EPOCHS,
                   help="Total training epochs.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    ensure_dirs()
    set_seed(config.RANDOM_SEED)

    # Override config if CLI flags were set
    if args.batch_size != config.BATCH_SIZE:
        config.BATCH_SIZE = args.batch_size
    if args.epochs != config.TOTAL_EPOCHS:
        config.TOTAL_EPOCHS = args.epochs

    setup_mixed_precision()

    data_dir = args.data_dir or config.DATA_DIR

    print("[train] Building tf.data pipelines…")
    # build_datasets returns: train_ds, val_ds, test_ds,
    #                         train_identities, val_identities, test_identities
    train_ds, val_ds, test_ds, class_names, val_ids, test_ids = build_datasets(
        data_dir=data_dir
    )
    num_classes = len(class_names)
    print(f"[train] Training on {num_classes} identities.")
    print(f"[train] Verification callback will use {len(val_ids)} val identities.")
# ── On-the-fly augmentation on the training set ────────────────────────
    from dataset import RandomOcclusionErasing

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.10),
        RandomOcclusionErasing(p=0.25),
    ], name="gpu_augmentation")

    # Properly structure train_ds for ArcFaceTrainer: ((images, labels), targets)
    train_ds = train_ds.map(
        lambda images, labels: (
            (data_augmentation(images, training=True), labels),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Properly structure val_ds to match ArcFaceTrainer's test_step expectation
    val_ds = val_ds.map(
        lambda images, labels: (
            (images, labels),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # ── Build model ────────────────────────────────────────────────────────
    print("[train] Building FaceResNet from scratch…")
    raw_full_model, embedding_model = build_model(num_classes=num_classes, training=True)
    full_model = ArcFaceTrainer(
        inputs  = raw_full_model.inputs,
        outputs = raw_full_model.outputs,
    )
    model_summary(full_model)

    # ── Resume logic ───────────────────────────────────────────────────────
    initial_epoch = 0
    if args.resume and os.path.exists(config.BEST_TRAIN_MODEL):
        print(f"[train] Resuming from {config.BEST_TRAIN_MODEL}…")
        loaded = tf.keras.models.load_model(
            config.BEST_TRAIN_MODEL,
            custom_objects={"ArcFaceLayer": __import__("arcface").ArcFaceLayer},
        )
        full_model = ArcFaceTrainer(inputs=loaded.inputs, outputs=loaded.outputs)
        csv_path   = os.path.join(config.LOG_DIR, "training_log.csv")
        if os.path.exists(csv_path):
            import csv
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                initial_epoch = int(rows[-1]["epoch"]) + 1
        print(f"[train] Resuming from epoch {initial_epoch}")

    # ── Train ──────────────────────────────────────────────────────────────
    history = train(
        full_model, embedding_model,
        train_ds, val_ds,
        data_dir, val_ids,
        initial_epoch=initial_epoch,
    )

    # ── Plot training curves ───────────────────────────────────────────────
    plot_training_history({"phase1": history.history}, save_dir=config.OUTPUT_DIR)

    # ── Final test-set verification ────────────────────────────────────────
    print("\n[train] Running final verification on test identities…")
    from evaluate import compute_verification_metrics

    test_paths1, test_paths2, test_labels = build_verification_pairs(
        data_dir   = data_dir,
        identities = test_ids,
    )

    all_paths    = test_paths1 + test_paths2
    unique_paths = list(dict.fromkeys(all_paths))
    p2i          = {p: i for i, p in enumerate(unique_paths)}

    def _load(path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((*config.IMAGE_SIZE, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        return img.astype(np.float32)

    embs = []
    for start in range(0, len(unique_paths), 64):
        batch = np.stack([_load(p) for p in unique_paths[start:start + 64]])
        embs.append(embedding_model.predict(batch, verbose=0))
    embs  = np.concatenate(embs, axis=0)
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

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
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