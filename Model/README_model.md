# VisiGuard — Model & Training Reference

## Architecture: FaceResNet

FaceResNet is a lightweight residual CNN designed specifically for 112×112 face images and trained entirely from scratch — no ImageNet weights, no pretrained backbone.

```
Input (112 × 112 × 3)
    │
    ▼
Stem:    Conv(64, 3×3, stride=2) → BN → PReLU          → 56×56×64
    │
    ▼
Stage 1: 2 × ResBlock(64,  stride=1)                   → 56×56×64
Stage 2: 2 × ResBlock(128, first stride=2)             → 28×28×128
Stage 3: 4 × ResBlock(256, first stride=2)             → 14×14×256
Stage 4: 2 × ResBlock(512, first stride=2)             →  7×7×512
    │
    ▼
GlobalAveragePooling2D → BN                            → 512
Dense(1024, no bias)   → BN → PReLU → Dropout(0.3)    → 1024
Dense(512,  no bias)   → BN → UnitNormalization        → 512-d embedding
    │
    ├─── Inference: cosine similarity matching
    │
    └─── Training only:
         ArcFaceLayer → logits (num_classes,)
```

### Residual Block

Each block follows the pre-activation style:

```
x → Conv(3×3) → BN → PReLU → Conv(3×3) → BN → Add(shortcut) → PReLU
```

The shortcut uses a 1×1 Conv + BN projection whenever spatial size or channel count changes. All convolutions use He (kaiming) initialisation to prevent vanishing/exploding gradients during from-scratch training.

### Why PReLU?

PReLU has a learnable negative slope per channel, which consistently outperforms ReLU for face recognition by allowing the network to encode subtle shading and shadow information that ReLU would discard.

### Embedding normalisation

The final `UnitNormalization` layer is forced to `dtype=float32` even under mixed-precision training. This prevents NaN values that occur when float16 underflows during the L2-norm computation on near-zero vectors early in training.

---

## Loss: ArcFace

ArcFace (Deng et al., CVPR 2019) adds a fixed angular margin *m* to the angle between an embedding and its true-class weight vector before computing the softmax:

```
logits[i] = s · cos(θᵢ + m)   if i == true class
logits[i] = s · cos(θᵢ)        otherwise
```

This directly optimises the angular separation between identities on the unit hypersphere rather than Euclidean distance, which maps well to cosine-similarity matching at inference.

### Hyperparameter choices for LFW scale

| Parameter | Value | Reason |
|-----------|-------|--------|
| Margin *m* | 0.5 (≈ 28.6°) | Paper default — unchanged |
| Scale *s* | 32 | Lowered from the paper's 64; with only ~8 K training images, s=64 makes the loss overconfident and impairs generalisation |
| L2 regulariser | 1e-4 | Tighter than fine-tuning scenarios to control overfitting on a small dataset |
| Dropout | 0.3 | Moderate; from-scratch training needs less aggressive regularisation than fine-tuning |

### Numerical stability

- `cos(θ)` is clipped to `[-1+ε, 1-ε]` before applying the margin.
- `sin(θ)` is derived via the Pythagorean identity — no arccos, no numerical issues near 0° or 180°.
- A linear fallback (`cos θ − sin(m)·m`) replaces `cos(θ+m)` when `θ+m > π` to prevent the angular penalty from wrapping around the antipodal point.

---

## Training Pipeline

### Single-phase from-scratch training

The model has **no frozen layers** and is fully trainable from epoch 0. The old two-phase (frozen warm-up → unfreeze) approach only applies when fine-tuning a pretrained backbone; it is a no-op here.

### Learning rate schedule

```
Epochs 0 → WARMUP_EPOCHS:    linear ramp 0 → INITIAL_LR
Epochs WARMUP_EPOCHS → 120:  cosine decay  INITIAL_LR → MIN_LR
```

| Parameter | Value |
|-----------|-------|
| `INITIAL_LR` | 1e-3 |
| `WARMUP_EPOCHS` | 10 |
| `TOTAL_EPOCHS` | 120 |
| `MIN_LR` | 1e-6 |
| `GRADIENT_CLIP_NORM` | 0.5 |

The warm-up ramp prevents large gradient spikes from random initial weights in the first few epochs. Gradient clipping at norm 0.5 provides a safety net throughout.

### Data augmentation

Augmentation is applied on-the-GPU inside the `tf.data` pipeline:

| Transform | Setting |
|-----------|---------|
| Horizontal flip | 50% probability |
| Random rotation | ±18° |
| Random zoom | ±12% |
| Brightness jitter | ±0.25 |
| Contrast jitter | ±0.25 |
| Saturation jitter | ±0.15 |
| Hue jitter | ±0.05 |
| Random occlusion erasing | 25% probability, patch 2–20% of image area |

Occlusion erasing simulates real-world partial occlusions (sunglasses, hands, masks) by replacing a random rectangle with uniform noise.

### Data splits

Splits are performed at the **identity level** — no person appears in more than one split:

| Split | Fraction | Role |
|-------|----------|------|
| Train | 80% | Classification + ArcFace |
| Val | 10% | `VerificationCallback` — pairwise AUC/EER per epoch |
| Test | 10% | Final evaluation after training ends |

### Training wrapper: `ArcFaceTrainer`

Standard Keras `.fit()` does not pass labels into the model `call()`. `ArcFaceTrainer` is a `tf.keras.Model` subclass that overrides `train_step` and `test_step` to inject integer labels into the `ArcFaceLayer` during both the forward pass and the metric computation.

### Callbacks

| Callback | Monitor | Action |
|----------|---------|--------|
| `VerificationCallback` | — | Computes pairwise AUC/EER from val identities each epoch; populates `val_ver_auc` and `val_ver_eer` in the logs |
| `ModelCheckpoint` (best) | `val_ver_auc ↑` | Saves full model to `checkpoints/best_train_model.keras` |
| `ModelCheckpoint` (epoch) | — | Saves weights to `checkpoints/epoch{N}_auc{AUC}.weights.h5` |
| `EarlyStopping` | `val_ver_auc ↑` | Stops after 20 epochs without improvement; restores best weights |
| `ReduceLROnPlateau` | `val_loss ↓` | Halves LR after 7 stagnant epochs, floor `MIN_LR` |
| `LearningRateScheduler` | — | Cosine schedule with warm-up |
| `TensorBoard` | — | Logs to `logs/training/` |
| `CSVLogger` | — | Appends per-epoch metrics to `logs/training_log.csv` |

### Mixed precision

When a Tensor Core GPU (compute capability ≥ 7.0 — T4, A100, RTX 30xx) is detected, training automatically runs in `mixed_float16` for approximately 2× throughput. The embedding normalisation layer is pinned to `float32` regardless, preventing precision-related NaNs.

---

## Evaluation Metrics

Face recognition is a **verification** problem ("are these the same person?"), not a classification problem. The evaluation pipeline computes:

| Metric | Description |
|--------|-------------|
| **AUC** | Area under the ROC curve — overall discriminability |
| **EER** | Equal Error Rate — threshold where FAR = FRR; lower is better |
| **TAR @ FAR=0.1%** | True Accept Rate at a very tight False Accept Rate |
| **TAR @ FAR=1%** | TAR at the standard operating point |
| **Optimal threshold** | F1-maximising cosine similarity cutoff |

Target performance on LFW (~1,680 identities, ≥5 images):

| Metric | Target |
|--------|--------|
| AUC | ≥ 0.95 |
| EER | ≤ 8% |
| TAR @ FAR=1% | ≥ 85% |

### Confidence calibration

Raw cosine similarity is mapped to a human-readable confidence percentage using a sigmoid centred on `SAME_PERSON_THRESHOLD` (0.55) with slope k=12:

```
confidence = sigmoid(12 · (sim − 0.55))
```

This ensures a score right at the decision boundary reads ~50%, not 75%.

---

## Inference

Two models are saved after training:

| File | Contents | Use |
|------|----------|-----|
| `checkpoints/best_train_model.keras` | Full model including ArcFace head | Resume training |
| `checkpoints/best_embedding_model.keras` | Embedding model only (512-d output) | All inference tasks |

### Inference modes (`inference.py`)

```bash
# 1:1 verification
python inference.py --mode compare --img1 alice.jpg --img2 bob.jpg

# 1:N identification against a gallery
python inference.py --mode identify --img group.jpg --gallery data/gallery/

# Live webcam (Google Colab)
python inference.py --mode webcam --gallery data/gallery/
```

### Gallery format

```
data/gallery/
    Alice_Smith/
        photo1.jpg
        photo2.jpg
    Bob_Jones/
        photo1.jpg
```

One sub-folder per identity. Multiple images per identity are embedded and averaged (then re-normalised) to produce a robust representative vector.

---

## File Map

| File | Role |
|------|------|
| `config.py` | All hyperparameters and paths — edit here first |
| `model.py` | FaceResNet architecture, LR scheduler, compile helper |
| `arcface.py` | ArcFace loss layer (`ArcFaceLayer`) |
| `train.py` | Training loop, `ArcFaceTrainer`, `VerificationCallback` |
| `dataset.py` | `build_datasets()`, augmentation, `build_verification_pairs()` |
| `evaluate.py` | ROC/AUC/EER/TAR metrics and plots |
| `inference.py` | `compare_two_images()`, `FaceIdentifier`, `run_webcam_colab()` |
| `detector.py` | YOLOv8-face wrapper (`FaceDetector`) |
| `aligner.py` | Eye-landmark affine alignment (`FaceAligner`) |
| `utils.py` | Shared helpers: mixed precision, plotting, gallery builder |
| `download_lfw.py` | Downloads LFW via TensorFlow Datasets |
| `training_demo.ipynb` | End-to-end Google Colab walkthrough |
