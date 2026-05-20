# 🔍 Face Recognition CNN — Production-Level Project

A complete, production-quality face recognition system using:

| Component | Choice | Why |
|-----------|--------|-----|
| **Backbone** | ResNet50V2 (ImageNet pretrained) | Proven for face recognition; stable gradient flow |
| **Loss** | ArcFace (Additive Angular Margin) | Tighter identity clusters → better generalisation |
| **Detector** | YOLOv8-face | State-of-the-art real-time face detection |
| **Framework** | TensorFlow / Keras | Mature ecosystem; easy GPU support |

---

## 📁 Project Structure

```
face_recognition_project/
├── config.py          ← All hyperparameters in one place
├── model.py           ← CNN architecture (backbone + ArcFace head)
├── arcface.py         ← ArcFace loss layer
├── dataset.py         ← tf.data pipeline with augmentation
├── detector.py        ← YOLOv8 face detector wrapper
├── train.py           ← Training script (2-phase)
├── inference.py       ← Compare faces / webcam / gallery lookup
├── utils.py           ← Helpers: plotting, cosine similarity, etc.
├── download_lfw.py    ← One-command dataset download
├── requirements.txt
├── README.md
├── checkpoints/       ← Model weights saved here
├── logs/              ← TensorBoard logs
├── data/
│   └── faces/         ← Training images (one folder per person)
├── outputs/           ← Training curves, result images
└── notebooks/
    └── training_demo.ipynb
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
# (Recommended) Create a virtual environment first
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **GPU users:** Make sure CUDA ≥ 11.2 and cuDNN ≥ 8.1 are installed.
> TensorFlow will detect the GPU automatically.

---

### 2. Download the dataset

#### Option A – LFW (automatic, recommended for beginners)

```bash
python download_lfw.py
```

This downloads the Labeled Faces in the Wild dataset (~180 MB), filters it
to identities with ≥5 images, and places everything under `data/faces/`.

**Result:** ~1,680 identities · ~10,000 images — ready to train.

#### Option B – Your own dataset

Create sub-folders under `data/faces/`, one per person:

```
data/faces/
    Alice_Smith/
        001.jpg
        002.jpg
        003.jpg
    Bob_Jones/
        001.jpg
        002.jpg
        ...
```

Each person must have at least **5 images** (configurable via `config.py`).

#### Option C – VGGFace2 (production-scale, 85%+ accuracy)

VGGFace2 has 3.3M images of 9,131 identities.

1. Request access at <https://github.com/ox-vgg/vgg_face2>
2. Download and organise as `data/faces/<identity>/<image.jpg>`
3. Run `python train.py`

---

### 3. Train the model

```bash
# Standard training
python train.py

# Resume from a previous run
python train.py --resume

# Override batch size (e.g., if GPU runs out of memory)
python train.py --batch-size 16
```

**Training has two phases:**

| Phase | Epochs | Backbone | LR | Purpose |
|-------|--------|----------|----|---------|
| Warm-up | 15 | Frozen | 1e-3 | Train head from scratch |
| Fine-tune | 35 | Top 50 layers unfrozen | 1e-4 → 1e-7 (cosine) | Adapt backbone |

**Watch training in TensorBoard:**

```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

---

### 4. Run inference

#### Compare two face images

```bash
python inference.py --mode compare --img1 alice.jpg --img2 bob.jpg
```

```
─── Face Comparison Result ─────────────────────────
  similarity     : 0.7832
  same_person    : True
  verdict        : Same Person ✓
  confidence     : 89.2%
  threshold      : 0.55
────────────────────────────────────────────────────
```

#### Identify faces in a photo

```bash
python inference.py \
  --mode identify \
  --img group_photo.jpg \
  --gallery data/gallery/ \
  --output outputs/result.jpg
```

#### Live webcam recognition (press Q to quit)

```bash
python inference.py --mode webcam --gallery data/gallery/
```

---

## 🎛️ Hyperparameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | `(112, 112)` | Input resolution (standard for face recognition) |
| `EMBEDDING_DIM` | `512` | Face embedding vector size |
| `DROPOUT_RATE` | `0.4` | Dropout in the bottleneck head |
| `ARCFACE_MARGIN` | `0.5` | Angular margin ≈ 28.6° (ArcFace paper default) |
| `ARCFACE_SCALE` | `64.0` | Feature scale / temperature |
| `BATCH_SIZE` | `32` | Training batch size |
| `WARMUP_EPOCHS` | `15` | Phase 1 epochs (backbone frozen) |
| `FINETUNE_EPOCHS` | `35` | Phase 2 epochs (top layers unfrozen) |
| `SAME_PERSON_THRESHOLD` | `0.55` | Cosine similarity cutoff for matching |
| `FACE_CONF_THRESHOLD` | `0.50` | YOLOv8 minimum face detection confidence |
| `MIXED_PRECISION` | `True` | float16 training (GPU only, 2× speedup) |

---

## 📊 Expected Accuracy

| Dataset | Identities | Validation Accuracy |
|---------|-----------|---------------------|
| LFW (≥5 imgs/person) | ~1,680 | **85 – 92%** |
| CASIA-WebFace | 10,575 | **90 – 95%** |
| VGGFace2 | 9,131 | **93 – 97%** |

> Validation accuracy here measures **classification** on the training identities.
> For **1:1 face verification** (same/different person test), the similarity
> threshold controls precision vs recall.

---

## 🏗️ Architecture Deep-Dive

```
Input (112 × 112 × 3)
    │
    ▼
ResNet50V2  [ImageNet pretrained — 25M params]
    │   ↑ Phase 1: frozen
    │   ↑ Phase 2: top 50 layers trainable
    ▼
GlobalAveragePooling2D   → (2048,)
BatchNormalization
    │
    ▼
Dense(1024, no_bias)
BatchNormalization
ReLU
Dropout(0.4)
    │
    ▼
Dense(512, no_bias, L2_reg=5e-4)
BatchNormalization
L2-Normalise  ────────────────────── Embedding (512,) for inference
    │
    ▼  [training only]
ArcFaceLayer
    • W: (512, num_classes) — one prototype per identity
    • cos(θ) = emb · W_norm
    • margin: cos(θ+m) for true class, cos(θ) for others
    • scale by 64.0
    │
    ▼
Logits (num_classes,)
SparseCategoricalCrossentropy loss
```

---

## 🔧 Common Issues & Fixes

**"No face detected"**
- Try lowering `FACE_CONF_THRESHOLD` in `config.py` (e.g., `0.35`)
- Make sure the image contains a clearly visible, frontal face
- Minimum detected face size is `20×20` pixels; use a higher-resolution image

**Out-of-memory (OOM) on GPU**
- Reduce `BATCH_SIZE` to `16` or `8`
- Disable mixed precision: `MIXED_PRECISION = False` in `config.py`

**Training loss is NaN**
- Gradient explosion — try reducing `WARMUP_LR` to `5e-4`
- Lower `ARCFACE_SCALE` from 64 to 32

**Low accuracy (< 70%)**
- Dataset too small: need at least 1,000 identities for good results
- Try increasing `MIN_IMAGES_PER_CLASS` to 10 for cleaner data
- Increase `FINETUNE_EPOCHS` to 50

**Model file not found at inference**
- Must run `python train.py` first
- Default path: `checkpoints/best_embedding_model.keras`

**YOLOv8 download fails**
- Manually download from [HuggingFace](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)
- Place `model.pt` at `checkpoints/yolov8n_face.pt`

---

## 📚 References

- **ArcFace**: Deng et al. (2019) — <https://arxiv.org/abs/1801.07698>
- **ResNet50V2**: He et al. (2016) — <https://arxiv.org/abs/1603.05027>
- **YOLOv8**: Ultralytics — <https://docs.ultralytics.com>
- **LFW Dataset**: <http://vis-www.cs.umass.edu/lfw/>
- **VGGFace2**: <https://github.com/ox-vgg/vgg_face2>

---

## 📜 Licence

MIT — free for personal and commercial use.
