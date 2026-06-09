# 🔍 VisiGuard — מערכת אבטחה חכמה עם זיהוי פנים בזמן אמת

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react&logoColor=black)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Face-00FFAA?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**מערכת אבטחה ביתית אקטיבית המזהה פנים בזמן אמת, מבדילה בין אנשים מורשים ולא מוכרים, ומתריעה מיידית למשתמש.**

</div>

---

## תוכן עניינים

- [מה זה VisiGuard?](#מה-זה-visiguard)
- [ארכיטקטורת המערכת](#ארכיטקטורת-המערכת)
- [זרימת הנתונים](#זרימת-הנתונים)
- [מודל ה-AI](#מודל-ה-ai)
- [ה-API / Backend](#ה-api--backend)
- [ממשק המשתמש Frontend](#ממשק-המשתמש-frontend)
- [Dataset](#dataset)
- [מדדי ביצועים](#מדדי-ביצועים)
- [התקנה והרצה](#התקנה-והרצה)
- [מבנה הפרויקט](#מבנה-הפרויקט)
- [אימון מחדש של המודל](#אימון-מחדש-של-המודל)

---

## מה זה VisiGuard?

רוב מצלמות האבטחה הביתיות מגיבות **אחרי** שמשהו קורה — הן מקליטות אך לא מבינות. VisiGuard לוקחת גישה שונה: **ניתוח בזמן אמת** של כל מי שמופיע בשדה הראייה של המצלמה.

**כיצד זה עובד:**
- מצלמה ממוקמת בנקודה אסטרטגית (כניסה לבית, שער, מסדרון)
- כל פריים מנותח על ידי YOLOv8 לזיהוי נוכחות אנושית
- אם מזוהה אדם — פניו מחולצות ומועברות למודל ה-CNN המאומן
- המודל משווה את ה-Embedding של הפנים מול גלריית האנשים המורשים
- אם האדם מזוהה — האירוע נרשם בשקט
- אם האדם **לא מזוהה** — מתריאה Push מיידית נשלחת לטלפון

---

## ארכיטקטורת המערכת

```
┌─────────────────────────────────────────────────────────────────┐
│                        VISIGUARD PIPELINE                       │
│                                                                 │
│  📷 Camera Feed                                                 │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐    לא נמצא אדם    ┌──────────────┐            │
│  │  YOLOv8     │ ─────────────────► │  המשך ניטור  │            │
│  │ Person Det. │                    └──────────────┘            │
│  └──────┬──────┘                                                │
│         │ נמצא אדם                                              │
│         ▼                                                       │
│  ┌─────────────┐                                                │
│  │  FaceAligner│  ← YOLOv8 5-point keypoints (eyes, nose, ...)  │
│  │ (eye warp)  │                                                │
│  └──────┬──────┘                                                │
│         ▼                                                       │
│  ┌─────────────────────────────┐                                │
│  │         FaceResNet          │                                │
│  │   CNN trained from scratch  │                                │
│  │   512-d L2 embedding        │                                │
│  └──────────────┬──────────────┘                                │
│                 ▼                                               │
│  ┌──────────────────────────────┐                               │
│  │       Decision Engine        │                               │
│  │  cosine_sim ≥ 0.55 → מורשה  │                               │
│  │  cosine_sim < 0.55 → לא מוכר│                               │
│  └──────┬───────────────┬───────┘                               │
│         │               │                                       │
│         ▼               ▼                                       │
│    ✅ רישום         🚨 התראה                                    │
│    שקט             Push Notification                            │
│                    + שמירת אירוע ב-DB                           │
│                         │                                       │
│                         ▼                                       │
│                  ┌─────────────┐                                │
│                  │  React App  │  Dashboard / Alerts / Gallery  │
│                  └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## זרימת הנתונים

```
[Camera] ──► [YOLOv8 Detect] ──► [FaceAligner] ──► [FaceResNet CNN]
                                                          │
                                                    512-d Embedding
                                                          │
                                              ┌───────────▼────────────┐
                                              │   Cosine Similarity     │
                                              │   vs. Gallery DB        │
                                              └───────────┬────────────┘
                                                          │
                                            ┌─────────────┴──────────────┐
                                            │                            │
                                      sim ≥ 0.55                  sim < 0.55
                                            │                            │
                                      ✅ Authorized              ❌ Unknown
                                       → Log Event            → Push Alert
                                                               → Save to DB
```

---

## מודל ה-AI

### FaceResNet — רשת מותאמת לפנים, מאומנת מאפס

המודל **לא** מבוסס על מודל מאומן-מראש (ImageNet, EfficientNet וכד'). הוא תוכנן ואומן מאפס ספציפית למשימת זיהוי פנים.

#### ארכיטקטורת הרשת

```
Input (112 × 112 × 3)
    │
    ▼
Stem:    Conv(64, 3×3, stride=2) → BN → PReLU       → 56×56×64
    │
    ▼
Stage 1: 2 × ResBlock(64,  stride=1)                → 56×56×64
Stage 2: 2 × ResBlock(128, first stride=2)          → 28×28×128
Stage 3: 4 × ResBlock(256, first stride=2)          → 14×14×256
Stage 4: 2 × ResBlock(512, first stride=2)          →  7×7×512
    │
    ▼
GlobalAveragePooling2D → BN                         → 512
Dense(1024, no bias)   → BN → PReLU → Dropout(0.3) → 1024
Dense(512,  no bias)   → BN → UnitNorm (float32)   → 512-d Embedding
    │
    ├── Inference: cosine similarity matching
    └── Training:  ArcFaceLayer → Logits(num_classes)
```

#### בלוק Residual

```python
x → Conv(3×3) → BN → PReLU → Conv(3×3) → BN → Add(shortcut) → PReLU
```

כל שינוי ב-stride או ב-channels מטופל על ידי projection shortcut (1×1 Conv + BN).

---

### ArcFace Loss

ArcFace מוסיף **מרווח זוויתי (m=0.5)** בין זהויות שונות במרחב ה-Embedding:

```
logits[i] = s · cos(θᵢ + m)    עבור המחלקה הנכונה
logits[i] = s · cos(θᵢ)         עבור שאר המחלקות
```

המימוש מחשב את `cos(θ+m)` ישירות דרך זהות טריגונומטרית (ללא arccos יקר):

```
cos(θ + m) = cos θ · cos m − sin θ · sin m
```

| פרמטר | ערך | הסיבה |
|-------|-----|--------|
| Margin m | 0.5 (≈28.6°) | ערך ברירת מחדל מהמאמר |
| Scale s | 32 | הורד מ-64 — עם ~8K תמונות, s=64 גורם לאימון להיות over-confident |
| L2 Regularizer | 1e-4 | מניעת overfitting על dataset קטן |
| Dropout | 0.3 | פחות אגרסיבי מ-fine-tuning |

---

### לוח זמנים האימון

| שלב | אפוקים | LR | מה קורה |
|-----|--------|----|---------| 
| Warm-up | 0–9 | 0 → 1e-3 (linear) | יציבות עם weights אקראיים |
| Cosine Decay | 10–120 | 1e-3 → 1e-6 | כיוון עדין |

**אסטרטגיית אימון:** כל השכבות trainable מאפוק 0 — אין Phase frozen.  
**Mixed Precision:** float16 אוטומטי על T4/A100 (~2× speedup). שכבת UnitNorm מקובעת ל-float32 למניעת NaN.

---

### Augmentation Pipeline

| טרנספורמציה | הגדרה |
|-------------|-------|
| Horizontal Flip | 50% |
| Random Rotation | ±18° |
| Random Zoom | ±12% |
| Brightness | ±0.25 |
| Contrast | ±0.25 |
| Saturation | ±0.15 |
| Hue | ±0.05 |
| Occlusion Erasing | 25% — מדמה חסימות חלקיות (משקפיים, יד, מסכה) |

---

### Callbacks

| Callback | Monitor | פעולה |
|----------|---------|-------|
| `VerificationCallback` | — | מחשב AUC/EER מ-val identities כל epoch |
| `ModelCheckpoint` (best) | `val_ver_auc ↑` | שומר `best_train_model.keras` |
| `EarlyStopping` | `val_ver_auc ↑` | עוצר אחרי 20 epochs ללא שיפור |
| `ReduceLROnPlateau` | `val_loss ↓` | מחצה LR אחרי 7 epochs |
| `TensorBoard` | — | logs ל-`logs/training/` |
| `CSVLogger` | — | `logs/training_log.csv` |

---

## ה-API / Backend

הבאקאנד בנוי ב-**Python** ומשמש כגשר בין צינור ה-AI לבין ה-Frontend.

### תפקידים עיקריים

- **קבלת פריימים מהמצלמה** ושליחתם לצינור הזיהוי
- **ניהול גלריית הפנים** — שמירת, עדכון ומחיקת embeddings
- **שמירת היסטוריית אירועים** — כל זיהוי (מורשה / לא מוכר) עם timestamp
- **שליחת התראות Push** לאפליקציית המשתמש בזמן אמת
- **חשיפת REST API** לממשק המשתמש

### מודולים מרכזיים

| קובץ | תפקיד |
|------|-------|
| `detector.py` | YOLOv8-face wrapper — זיהוי + 5-point keypoints |
| `aligner.py` | יישור עין-לעין (affine warp) לפני CNN |
| `inference.py` | `compare_two_images()`, `FaceIdentifier`, `run_webcam_colab()` |
| `evaluate.py` | ROC / AUC / EER / TAR@FAR |
| `utils.py` | mixed precision, gallery builder, training curves |
| `config.py` | כל ה-hyperparameters והנתיבים — עריכה ממקום אחד |

### מצבי Inference

```bash
# השוואת שתי תמונות (1:1 verification)
python inference.py --mode compare --img1 alice.jpg --img2 bob.jpg

# זיהוי בתמונה קבוצתית מול גלריה (1:N identification)
python inference.py --mode identify --img group.jpg --gallery data/gallery/

# מצלמה חיה (Google Colab)
python inference.py --mode webcam --gallery data/gallery/
```

### פורמט תגובת Verification

```json
{
  "similarity": 0.7283,
  "confidence": 0.89,
  "match": true,
  "verdict": "Same Person ✓"
}
```

### Confidence Calibration

דמיון קוסינוס גולמי ממופה לאחוז ביטחון קריא דרך sigmoid:

```python
confidence = sigmoid(12 · (cosine_sim − 0.55))
```

ציון בדיוק על הסף (0.55) יוצג כ-50%, לא 75%.

### Gallery — מבנה התיקיות

```
data/gallery/
    Alice_Smith/
        photo1.jpg
        photo2.jpg     ← ממוצע embeddings לוקטור אחד יציב
    Bob_Jones/
        photo1.jpg
```

---

## ממשק המשתמש Frontend

הממשק בנוי ב-**React** ומהווה את מרכז השליטה של המערכת.

### מסכים

| מסך | תיאור |
|-----|-------|
| **Login / Register** | כניסה והרשמה עם שם משתמש וסיסמה |
| **Dashboard** | מצב שרת, מצב ניטור, הודעות סטטוס — נקודת מוצא לכל המסכים |
| **Monitoring** | שידור חי מהמצלמה + זיהוי פנים בזמן אמת. מתג הפעלה/כיבוי עם חיווי צבעוני. פועל ברקע בזמן מעבר בין מסכים |
| **Gallery** | ניהול אנשים מורשים — העלאת תמונות, שיוך שמות, מחיקה, עדכון |
| **Security Alerts** | היסטוריית כל האירועים החריגים עם timestamp וסטטוס טיפול |
| **Cameras** | בחירת מצלמה פעילה + תצוגה מקדימה לפני הפעלה |
| **Face Comparison** | בדיקה ידנית של התאמה בין שתי תמונות — לצורכי בדיקה ופיתוח |
| **Legacy Alerts** | תאימות לאחור עם התראות ישנות |

### תכונות עיקריות

- **Push Notifications** — התראה מיידית כשמזוהה אדם לא מוכר
- **Real-time Status** — חיווי חי של מצב המערכת (ניטור פעיל / כבוי)
- **Gallery Management** — ממשק לניהול מלא של אנשים מורשים
- **Event Log** — כל אירוע מתועד עם זמן, תמונה וסטטוס

---

## Dataset

הפרויקט מאומן על **LFW (Labeled Faces in the Wild)** — דאטאסט פנים מהעולם האמיתי, בלתי-מבוקר.

### הכנת הנתונים

| שלב | פעולה |
|-----|-------|
| הורדה | `python download_lfw.py --min-images 5` — דרך TensorFlow Datasets |
| סינון | נשמרות רק זהויות עם ≥5 תמונות (~1,680 זהויות) |
| יישור | `prepare_aligned_dataset()` — YOLOv8 + eye-warp לכל תמונה |
| חלוקה | **Identity-level split** — 80% train / 10% val / 10% test, ללא data leakage |

> **למה identity-level split?**  
> חלוקה אקראית של תמונות תאפשר לאותה פנים להופיע גם ב-train וגם ב-test — מה שייצור מדדים מנופחים שלא מייצגים יכולת הכללה אמיתית.

### זוגות אימות (Verification Pairs)

`build_verification_pairs()` יוצרת זוגות מאוזנים מ-val/test בלבד:
- 50% זוגות חיוביים (אותו אדם)
- 50% זוגות שליליים (אנשים שונים)

זה מאפשר מדידת ROC/EER אמיתית, ולא סיווג רגיל.

---

## מדדי ביצועים

### תוצאות ROC על LFW

| מדד | ערך |
|-----|-----|
| **AUC** | 0.90 |
| **EER** | 18.96% |
| **TAR @ FAR=0.1%** | 22.4% |
| **TAR @ FAR=1.0%** | 22.4% |
| **TAR @ FAR=10.0%** | 72.1% |

### בדיקה עצמאית

| השוואה | Similarity Score | תוצאה |
|--------|-----------------|-------|
| אותה תמונה | 1.000 | 100% match |
| אדם מול אישה שונה | 0.728 | המודל הבחין בשוני |
| שני גברים שונים | 0.334 | זיהוי חד-משמעי כשונים |
| שני אחים (דמיון גנטי) | 0.764 | הבחין בשוני, אך ציון גבוה יחסית — צפוי |

---

## התקנה והרצה

### דרישות מוקדמות

- Python 3.10+
- Node.js 18+ (לפרונטאנד)
- GPU עם CUDA (מומלץ — T4/A100 לאימון)

### Backend

```bash
# שכפול הריפו
git clone https://github.com/YOUR_USERNAME/VisiGuard
cd VisiGuard

# התקנת תלויות Python
pip install tensorflow ultralytics huggingface_hub scikit-learn \
            opencv-python tensorflow-datasets

# הורדת הנתונים
python download_lfw.py --min-images 5

# יישור פנים
python -c "from dataset import prepare_aligned_dataset; prepare_aligned_dataset()"

# אימון המודל (Google Colab מומלץ — ראה training_demo.ipynb)
python train.py

# הערכה
python evaluate.py

# הרצת זיהוי
python inference.py --mode compare --img1 face1.jpg --img2 face2.jpg
```

### Frontend

```bash
cd frontend
npm install
npm start
# האפליקציה תרוץ על http://localhost:3000
```

### Google Colab — אימון מלא

```
Runtime → Change runtime type → GPU (T4 or A100)
```

פתח את `training_demo.ipynb` ורוץ תא-תא. הנוטבוק מכסה:
1. בדיקת GPU
2. התקנת תלויות (כולל `tensorflow-datasets`)
3. הורדת LFW
4. יישור פנים
5. אימון (~20–40 דקות על T4)
6. הצגת עקומות אימון
7. הערכה (ROC + similarity distributions)
8. Inference

---

## מבנה הפרויקט

```
VisiGuard/
│
├── 📂 AI Model
│   ├── model.py           # FaceResNet architecture
│   ├── arcface.py         # ArcFace loss layer
│   ├── train.py           # Training pipeline (ArcFaceTrainer + callbacks)
│   ├── dataset.py         # Data loading, augmentation, verification pairs
│   ├── evaluate.py        # ROC / AUC / EER metrics + plots
│   ├── inference.py       # compare, identify, webcam modes
│   ├── detector.py        # YOLOv8-face wrapper
│   ├── aligner.py         # Eye-landmark affine alignment
│   ├── download_lfw.py    # LFW via TensorFlow Datasets
│   ├── config.py          # ← כל ה-hyperparameters כאן
│   └── utils.py           # Mixed precision, plotting, gallery builder
│
├── 📒 training_demo.ipynb # Colab walkthrough — end to end
│
├── 📂 data/
│   ├── faces/             # Raw LFW images (after download_lfw.py)
│   ├── faces_aligned/     # Aligned crops (after prepare_aligned_dataset)
│   └── gallery/           # Authorized identities for runtime matching
│
├── 📂 checkpoints/
│   ├── best_embedding_model.keras  # Inference-only (512-d output)
│   ├── best_train_model.keras      # Full model incl. ArcFace head
│   └── yolov8n_face.pt             # YOLOv8-face weights (auto-downloaded)
│
├── 📂 outputs/
│   ├── training_history.png
│   ├── roc_curve.png
│   ├── similarity_dist.png
│   └── eval_metrics.json
│
├── 📂 logs/
│   ├── training/          # TensorBoard logs
│   └── training_log.csv
│
├── 📂 frontend/           # React application
│   ├── src/
│   │   ├── screens/       # Login, Dashboard, Monitoring, Gallery,
│   │   │                  # Alerts, Cameras, FaceComparison
│   │   └── components/
│   └── package.json
│
└── README.md
```

---

## אימון מחדש של המודל

### הגדרות מרכזיות ב-`config.py`

```python
# נתיבים
DATA_DIR   = '/content/FaceRecognition/data'

# ארכיטקטורה
EMBEDDING_DIM   = 512
DROPOUT_RATE    = 0.3
L2_REGULARIZER  = 1e-4

# ArcFace
ARCFACE_MARGIN  = 0.5
ARCFACE_SCALE   = 32.0

# אימון
BATCH_SIZE      = 64
TOTAL_EPOCHS    = 120
WARMUP_EPOCHS   = 10
INITIAL_LR      = 1e-3
MIN_LR          = 1e-6

# Inference
SAME_PERSON_THRESHOLD = 0.55
```

### המשך אימון מ-checkpoint

```bash
python train.py --resume
```

### שינוי גודל batch לפי GPU

```bash
python train.py --batch-size 32
```

---

## טכנולוגיות

| קטגוריה | טכנולוגיה |
|---------|-----------|
| שפת תכנות | Python 3.10 |
| Deep Learning | TensorFlow / Keras |
| זיהוי אנשים | YOLOv8 (Ultralytics) |
| זיהוי פנים | FaceResNet + ArcFace (מאפס) |
| Computer Vision | OpenCV |
| Dataset | LFW via TensorFlow Datasets |
| Frontend | React |
| מטריקות | scikit-learn (ROC, AUC, EER) |
| ניהול קוד | Git / GitHub (branch-based workflow) |

---

## רפרנסים

- Deng et al. — [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698), CVPR 2019
- [YOLOv8-face — arnabdhar/YOLOv8-Face-Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)
- [LFW Dataset — Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
- [TensorFlow Datasets — LFW](https://www.tensorflow.org/datasets/catalog/lfw)

---

<div align="center">

פותח על ידי **נועם כהן** | VisiGuard — שכבת הגנה אקטיבית לבית החכם

</div>
