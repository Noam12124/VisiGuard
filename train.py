"""
train.py — Two-phase training script for the face recognition model.

Usage (Colab / command line):

    python train.py                    # fresh training
    python train.py --resume           # resume from latest checkpoint
    python train.py --batch-size 16    # override batch size (OOM fix)
    python train.py --phase 2          # jump straight to fine-tuning

Phase 1 — Warm-up (backbone frozen):
    • Only the embedding head + ArcFace layer are trained.
    • LR: cosine decay from WARMUP_LR.
    • Prevents large gradient updates from destroying pretrained features.

Phase 2 — Fine-tuning (top UNFREEZE_TOP_LAYERS of backbone unfrozen):
    • BatchNorm layers remain frozen (prevents covariate shift).
    • LR: cosine decay from FINETUNE_LR (10× smaller).
    • Gradient clipping: clipnorm=1.0.
"""

import os
import sys
import argparse
import json
import math

import tensorflow as tf
import numpy as np

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
    prepare_aligned_dataset,
)
from utils import (
    plot_training_history,
    setup_mixed_precision,
    ensure_dirs,
    set_seed,
)


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train face recognition model.")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume training from latest checkpoint.")
    parser.add_argument("--phase",      type=int, default=1, choices=[1, 2],
                        help="Start from phase 1 (warm-up) or 2 (fine-tune).")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--data-dir",   type=str, default=None)
    parser.add_argument("--no-align",   action="store_true",
                        help="Skip offline alignment pass.")
    return parser.parse_args()


# ── Custom training loop (needed for ArcFace label injection) ─────────────

class ArcFaceTrainer(tf.keras.Model):
    """
    Thin wrapper that passes labels into the ArcFace layer during the
    forward pass.  Keras's standard .fit() doesn't support passing part
    of the input batch to a specific layer at call-time, so we override
    train_step / test_step.

    The dataset yields  ((image, label), label)  batches:
      - inputs[0] = image tensor
      - inputs[1] = integer label tensor (passed to ArcFace)
      - targets   = integer label tensor (used for loss)
    """

    def train_step(self, data):
        (images, labels), _ = data    # labels used twice

        with tf.GradientTape() as tape:
            logits = self([images, labels], training=True)
            loss   = self.compiled_loss(labels, logits)
            loss  += sum(self.losses)   # regularisation

            if self.dtype_policy.compute_dtype == "float16":
                scaled = self.optimizer.get_scaled_loss(loss)

        if self.dtype_policy.compute_dtype == "float16":
            grads = tape.gradient(scaled, self.trainable_variables)
            grads = self.optimizer.get_unscaled_gradients(grads)
        else:
            grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels), _ = data
        logits = self([images, labels], training=False)
        self.compiled_loss(labels, logits)
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}


def _wrap_as_arcface_trainer(full_model) -> ArcFaceTrainer:
    """Re-wrap the Functional model as ArcFaceTrainer (subclassed model)."""
    # Pull out the functional graph and re-expose as subclass
    trainer = ArcFaceTrainer(
        inputs  = full_model.inputs,
        outputs = full_model.outputs,
        name    = full_model.name,
    )
    return trainer


# ── Phase 1: Warm-up ───────────────────────────────────────────────────────

def train_phase1(
    full_model,
    embedding_model,
    train_ds,
    val_ds,
    num_classes: int,
    batch_size:  int,
) -> tf.keras.callbacks.History:
    print("\n" + "="*60)
    print("  PHASE 1 — WARM-UP  (backbone frozen)")
    print("="*60)

    freeze_backbone(full_model)

    lr_schedule = get_cosine_scheduler(
        base_lr       = config.WARMUP_LR,
        total_epochs  = config.WARMUP_EPOCHS,
        warmup_epochs = 3,          # 3-epoch linear ramp-up
        min_lr        = config.MIN_LR,
    )

    compile_model(full_model, lr=config.WARMUP_LR)

    callbacks = _build_callbacks(
        phase       = 1,
        lr_schedule = lr_schedule,
    )

    history = full_model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = config.WARMUP_EPOCHS,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # Save embedding model (inference weights)
    embedding_model.save(config.BEST_EMBEDDING_MODEL)
    print(f"[train] Phase 1 done. Embedding model saved → {config.BEST_EMBEDDING_MODEL}")

    return history


# ── Phase 2: Fine-tuning ───────────────────────────────────────────────────

def train_phase2(
    full_model,
    embedding_model,
    train_ds,
    val_ds,
    initial_epoch: int = 0,
) -> tf.keras.callbacks.History:
    print("\n" + "="*60)
    print("  PHASE 2 — FINE-TUNING  (top backbone layers unfrozen)")
    print("="*60)

    unfreeze_top_layers(full_model)

    lr_schedule = get_cosine_scheduler(
        base_lr      = config.FINETUNE_LR,
        total_epochs = config.WARMUP_EPOCHS + config.FINETUNE_EPOCHS,
        min_lr       = config.MIN_LR,
    )

    compile_model(full_model, lr=config.FINETUNE_LR)

    callbacks = _build_callbacks(
        phase        = 2,
        lr_schedule  = lr_schedule,
        initial_epoch = initial_epoch,
    )

    history = full_model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = config.WARMUP_EPOCHS + config.FINETUNE_EPOCHS,
        initial_epoch   = initial_epoch,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # Save final embedding model
    embedding_model.save(config.BEST_EMBEDDING_MODEL)
    print(f"[train] Phase 2 done. Embedding model saved → {config.BEST_EMBEDDING_MODEL}")

    return history


# ── Callbacks ──────────────────────────────────────────────────────────────

def _build_callbacks(
    phase:        int,
    lr_schedule,
    initial_epoch: int = 0,
) -> list:
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR,        exist_ok=True)

    ckpt_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"phase{phase}_epoch{{epoch:03d}}_val{{val_accuracy:.4f}}.keras"
    )

    callbacks = [
        # Save best checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath          = config.BEST_TRAIN_MODEL,
            monitor           = "val_accuracy",
            save_best_only    = True,
            save_weights_only = False,
            verbose           = 1,
        ),
        # Also save periodic checkpoints
        tf.keras.callbacks.ModelCheckpoint(
            filepath          = ckpt_path,
            save_best_only    = False,
            save_weights_only = True,
            verbose           = 0,
            save_freq         = "epoch",
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = config.EARLY_STOPPING_PATIENCE,
            restore_best_weights = True,
            verbose              = 1,
        ),
        # ReduceLROnPlateau (safety net alongside cosine schedule)
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = config.REDUCE_LR_FACTOR,
            patience = config.REDUCE_LR_PATIENCE,
            min_lr   = config.MIN_LR,
            verbose  = 1,
        ),
        # Cosine LR
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir         = os.path.join(config.LOG_DIR, f"phase{phase}"),
            histogram_freq  = 0,
            update_freq     = "epoch",
        ),
        # CSV logger (for offline analysis)
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.LOG_DIR, f"phase{phase}_log.csv"),
            append = (initial_epoch > 0),
        ),
    ]
    return callbacks


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    ensure_dirs()
    set_seed(config.RANDOM_SEED)

    # Mixed precision
    use_mp = setup_mixed_precision()

    # ── Dataset ────────────────────────────────────────────────────────────
    data_dir = args.data_dir or config.DATA_DIR

    if not args.no_align:
        print("[train] Running offline face alignment…")
        aligned_dir = prepare_aligned_dataset(src_dir=data_dir)
        data_dir = aligned_dir

    print("[train] Building tf.data pipelines…")
    train_ds, val_ds, test_ds, class_names = build_datasets(
        data_dir   = data_dir,
        batch_size = args.batch_size,
    )
    num_classes = len(class_names)
    print(f"[train] {num_classes} identities.")

    # ── Build model ────────────────────────────────────────────────────────
    print("[train] Building model…")
    full_model, embedding_model = build_model(num_classes=num_classes, training=True)

    if use_mp:
        # Wrap optimizer in LossScaleOptimizer for float16
        # (compile_model does this automatically when mixed precision is on)
        pass

    model_summary(full_model)

    # ── Resume logic ───────────────────────────────────────────────────────
    start_phase   = args.phase
    initial_epoch = 0

    if args.resume and os.path.exists(config.BEST_TRAIN_MODEL):
        print(f"[train] Resuming from {config.BEST_TRAIN_MODEL}")
        full_model = tf.keras.models.load_model(
            config.BEST_TRAIN_MODEL,
            custom_objects={"ArcFaceLayer": __import__("arcface").ArcFaceLayer},
        )
        # Try to infer which epoch we're at from the CSV log
        csv_p1 = os.path.join(config.LOG_DIR, "phase1_log.csv")
        csv_p2 = os.path.join(config.LOG_DIR, "phase2_log.csv")
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
        print(f"[train] Starting from phase {start_phase}, epoch {initial_epoch}")

    # ── Training ───────────────────────────────────────────────────────────
    histories = {}

    if start_phase == 1:
        h1 = train_phase1(full_model, embedding_model, train_ds, val_ds, num_classes, args.batch_size)
        histories["phase1"] = h1.history
        start_phase   = 2
        initial_epoch = config.WARMUP_EPOCHS

    if start_phase == 2:
        h2 = train_phase2(full_model, embedding_model, train_ds, val_ds,
                          initial_epoch=initial_epoch)
        histories["phase2"] = h2.history

    # ── Plot & save ────────────────────────────────────────────────────────
    plot_training_history(histories, save_dir=config.OUTPUT_DIR)

    # ── Quick test evaluation ──────────────────────────────────────────────
    print("\n[train] Evaluating on held-out test set…")
    compile_model(full_model, lr=config.MIN_LR)   # just to have metrics
    results = full_model.evaluate(test_ds, verbose=1)
    metrics = dict(zip(full_model.metrics_names, results))
    print(f"[train] Test results: {metrics}")

    # Save metrics
    with open(os.path.join(config.OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print("\n[train] Training complete.")
    print(f"        Embedding model: {config.BEST_EMBEDDING_MODEL}")
    print(f"        Outputs:         {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
