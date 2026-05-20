"""
dataset.py — tf.data pipeline for face recognition training.

Folder layout expected (one sub-folder per identity):
    data/faces/
        Alice_Smith/
            001.jpg  002.jpg  ...
        Bob_Jones/
            001.jpg  002.jpg  ...
        ...

Design notes:
  • Images are loaded from disk, aligned offline (see prepare_aligned_dataset),
    then fed through a pure TF augmentation pipeline for speed.
  • Offline alignment: call `prepare_aligned_dataset()` once before training.
    It writes pre-aligned copies to data/faces_aligned/ so every epoch reads
    the same, already-correct crops (alignment is deterministic; doing it
    on-the-fly in tf.data adds latency without benefit).
  • Augmentation is applied ONLY during training (not validation/test).
  • Returns integer class labels; ArcFace uses SparseCategoricalCrossentropy.

Dataset choice: CASIA-WebFace / LFW (bundled auto-download).
  • VGGFace2 is the best option (3.3M images, 9k identities) but requires
    manual access request.  Instructions in README.md.
  • CASIA-WebFace (500k images, 10k identities) is automatically downloadable
    and gives strong results.  We provide a Colab download cell in the notebook.
  • LFW is used ONLY for verification evaluation, not training.
"""

import os
import json
import shutil
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
import config
from aligner import FaceAligner


# ── Offline alignment pass ─────────────────────────────────────────────────

def prepare_aligned_dataset(
    src_dir:  str = config.DATA_DIR,
    dst_dir:  str = None,
    aligner:  FaceAligner = None,
) -> str:
    """
    Walk src_dir, detect+align every face with MediaPipe (fast, CPU),
    and write the result to dst_dir.

    NOTE: This does NOT use YOLOv8 for the training data to keep things
    fast.  Instead it uses a lightweight eye-detection approach:
      1. OpenCV Haar cascades for eyes (fast).
      2. Fallback to plain centre-crop if no eyes found.

    YOLOv8 is used at inference time (real images may have multiple faces,
    backgrounds, etc.).  Training images are typically already-cropped faces
    (CASIA, VGGFace2), so the main concern is rotation correction.

    Args:
        src_dir: Root folder with identity sub-directories.
        dst_dir: Output folder.  Defaults to src_dir + "_aligned".

    Returns:
        Path to the aligned dataset root.
    """
    if dst_dir is None:
        dst_dir = src_dir.rstrip("/\\") + "_aligned"

    if os.path.exists(dst_dir):
        print(f"[dataset] Aligned dataset already exists at {dst_dir}. Skipping.")
        return dst_dir

    if aligner is None:
        aligner = FaceAligner()

    # Load Haar cascade for eye detection (bundled with OpenCV)
    eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    identities = sorted([
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    ])

    total_written = 0
    total_skipped = 0

    for identity in identities:
        in_folder  = os.path.join(src_dir,  identity)
        out_folder = os.path.join(dst_dir, identity)
        os.makedirs(out_folder, exist_ok=True)

        image_files = [
            f for f in os.listdir(in_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        for fname in image_files:
            src_path = os.path.join(in_folder, fname)
            dst_path = os.path.join(out_folder, fname)

            if os.path.exists(dst_path):
                total_written += 1
                continue

            img_bgr = cv2.imread(src_path)
            if img_bgr is None:
                total_skipped += 1
                continue

            # Try to detect eyes and align
            gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            eyes  = eye_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )

            if len(eyes) >= 2:
                # Sort eyes by x-position (left to right)
                eyes = sorted(eyes, key=lambda e: e[0])
                # Eye centre = top-left + w/2, top + h/2
                left_eye_centre  = (
                    int(eyes[0][0] + eyes[0][2] / 2),
                    int(eyes[0][1] + eyes[0][3] / 2),
                )
                right_eye_centre = (
                    int(eyes[1][0] + eyes[1][2] / 2),
                    int(eyes[1][1] + eyes[1][3] / 2),
                )
                aligned = aligner.align(img_bgr, left_eye_centre, right_eye_centre)
            else:
                # Fallback: centre-crop + resize
                h, w = img_bgr.shape[:2]
                bbox = (0, 0, w, h)
                aligned = aligner.crop_and_resize(img_bgr, bbox)

            cv2.imwrite(dst_path, aligned)
            total_written += 1

        # Progress
        if total_written % 1000 == 0 and total_written > 0:
            print(f"  [align] Processed {total_written:,} images …", end="\r")

    print(f"\n[dataset] Alignment done. "
          f"Written: {total_written:,}  Skipped: {total_skipped:,}")
    print(f"[dataset] Aligned dataset at: {dst_dir}")
    return dst_dir


# ── Class catalogue ────────────────────────────────────────────────────────

def build_class_catalogue(
    data_dir: str,
    min_images: int = config.MIN_IMAGES_PER_CLASS,
) -> tuple[list[str], dict[str, int]]:
    """
    Scan data_dir and return:
      (class_names,  class_to_idx)

    Only identities with at least `min_images` are included.
    """
    identities = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    class_names = []
    for identity in identities:
        folder = os.path.join(data_dir, identity)
        n_imgs = sum(
            1 for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        )
        if n_imgs >= min_images:
            class_names.append(identity)

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print(f"[dataset] Found {len(class_names)} identities "
          f"(filtered from {len(identities)}, min={min_images}).")
    return class_names, class_to_idx


def save_class_catalogue(
    class_names: list[str],
    path: str = None,
) -> None:
    if path is None:
        path = os.path.join(config.CHECKPOINT_DIR, "class_names.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[dataset] Class catalogue saved to {path}")


def load_class_catalogue(path: str = None) -> list[str]:
    if path is None:
        path = os.path.join(config.CHECKPOINT_DIR, "class_names.json")
    with open(path) as f:
        return json.load(f)


# ── File list builder ──────────────────────────────────────────────────────

def build_file_list(
    data_dir: str,
    class_names: list[str],
    class_to_idx: dict[str, int],
) -> tuple[list[str], list[int]]:
    """
    Return (file_paths, labels) lists for all images in data_dir
    belonging to the given class catalogue.
    """
    file_paths = []
    labels     = []

    for name in class_names:
        folder = os.path.join(data_dir, name)
        idx    = class_to_idx[name]
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                file_paths.append(os.path.join(folder, fname))
                labels.append(idx)

    return file_paths, labels


def split_dataset(
    file_paths: list[str],
    labels: list[int],
    val_split:  float = config.VALIDATION_SPLIT,
    test_split: float = config.TEST_SPLIT,
    seed: int         = config.RANDOM_SEED,
) -> tuple:
    """
    Stratified split → (train, val, test) tuples of (paths, labels).
    """
    test_frac = test_split / (1.0 - val_split)

    paths_tr, paths_tmp, lbl_tr, lbl_tmp = train_test_split(
        file_paths, labels,
        test_size    = val_split + test_split,
        random_state = seed,
        stratify     = labels,
    )
    paths_val, paths_te, lbl_val, lbl_te = train_test_split(
        paths_tmp, lbl_tmp,
        test_size    = test_frac,
        random_state = seed,
        stratify     = lbl_tmp,
    )

    print(f"[dataset] Split: train={len(lbl_tr):,}  "
          f"val={len(lbl_val):,}  test={len(lbl_te):,}")
    return (paths_tr, lbl_tr), (paths_val, lbl_val), (paths_te, lbl_te)


# ── tf.data pipeline ───────────────────────────────────────────────────────

def _load_and_decode(path: tf.Tensor) -> tf.Tensor:
    """Read, decode, and convert to float32 in [0, 255]."""
    raw  = tf.io.read_file(path)
    img  = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img  = tf.image.resize(img, config.IMAGE_SIZE, method="bilinear")
    img  = tf.cast(img, tf.float32)
    return img


@tf.function
def _augment(image: tf.Tensor) -> tf.Tensor:
    """Apply random augmentation for training."""
    # Random horizontal flip
    if config.AUGMENT_FLIP:
        image = tf.image.random_flip_left_right(image)

    # Colour jitter
    image = tf.image.random_brightness(image, config.AUGMENT_BRIGHTNESS)
    image = tf.image.random_contrast(image, 1.0 - config.AUGMENT_CONTRAST,
                                             1.0 + config.AUGMENT_CONTRAST)
    image = tf.image.random_saturation(image, 1.0 - config.AUGMENT_SATURATION,
                                               1.0 + config.AUGMENT_SATURATION)
    image = tf.image.random_hue(image, config.AUGMENT_HUE)

    # Random rotation (±15°) via affine warp
    angle_rad = tf.random.uniform(
        [], -config.AUGMENT_ROTATION, config.AUGMENT_ROTATION
    ) * (3.14159265 / 180.0)
    image = _rotate(image, angle_rad)

    # Random zoom
    zoom = tf.random.uniform([], 1.0 - config.AUGMENT_ZOOM, 1.0 + config.AUGMENT_ZOOM)
    h, w = config.IMAGE_SIZE
    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) / zoom), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) / zoom), tf.int32)
    new_h = tf.clip_by_value(new_h, 1, h)
    new_w = tf.clip_by_value(new_w, 1, w)
    image = tf.image.resize_with_crop_or_pad(image, new_h, new_w)
    image = tf.image.resize(image, [h, w])

    image = tf.clip_by_value(image, 0.0, 255.0)
    return image


def _rotate(image: tf.Tensor, angle_rad: tf.Tensor) -> tf.Tensor:
    """Rotate image by angle (radians) using tfa or manual affine."""
    try:
        import tensorflow_addons as tfa
        return tfa.image.rotate(image, angle_rad, interpolation="BILINEAR")
    except ImportError:
        # Fallback: no rotation if tfa not available
        return image


def _make_dataset(
    file_paths: list[str],
    labels:     list[int],
    augment:    bool,
    shuffle:    bool,
    batch_size: int = config.BATCH_SIZE,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset for (image, image, label) batches.

    The model takes ([image, label], label) because ArcFace needs the
    label inside the forward pass.
    """
    paths_ds  = tf.data.Dataset.from_tensor_slices(file_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(labels, tf.int32)
    )
    ds = tf.data.Dataset.zip((paths_ds, labels_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(file_paths), 50_000), seed=config.RANDOM_SEED)

    def load_and_prep(path, label):
        img = _load_and_decode(path)
        if augment:
            img = _augment(img)
        return (img, label), label      # inputs = (image, label), target = label

    ds = ds.map(load_and_prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(
    data_dir:   str  = None,
    batch_size: int  = config.BATCH_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Full pipeline: align → catalogue → split → tf.data.

    Returns:
        train_ds, val_ds, test_ds, class_names
    """
    if data_dir is None:
        # Use aligned version if it exists, otherwise raw.
        aligned = config.DATA_DIR.rstrip("/\\") + "_aligned"
        data_dir = aligned if os.path.exists(aligned) else config.DATA_DIR

    class_names, class_to_idx = build_class_catalogue(data_dir)
    save_class_catalogue(class_names)

    file_paths, labels = build_file_list(data_dir, class_names, class_to_idx)

    (tr_p, tr_l), (va_p, va_l), (te_p, te_l) = split_dataset(file_paths, labels)

    train_ds = _make_dataset(tr_p, tr_l, augment=True,  shuffle=True,  batch_size=batch_size)
    val_ds   = _make_dataset(va_p, va_l, augment=False, shuffle=False, batch_size=batch_size)
    test_ds  = _make_dataset(te_p, te_l, augment=False, shuffle=False, batch_size=batch_size)

    return train_ds, val_ds, test_ds, class_names


# ── Verification pairs (for ROC / EER evaluation) ─────────────────────────

def build_verification_pairs(
    data_dir:           str   = None,
    pairs_per_identity: int   = config.EVAL_PAIRS_PER_CLASS,
    seed:               int   = config.RANDOM_SEED,
) -> tuple[list, list, list[int]]:
    """
    Build (path1, path2, is_same) triples for verification evaluation.

    Genuine pairs: two different images of the same person.
    Impostor pairs: two images of different people.
    Returns:  paths1, paths2, labels  (1=same, 0=different)
    """
    if data_dir is None:
        aligned = config.DATA_DIR.rstrip("/\\") + "_aligned"
        data_dir = aligned if os.path.exists(aligned) else config.DATA_DIR

    rng   = np.random.default_rng(seed)
    class_images: dict[str, list[str]] = {}

    for identity in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, identity)
        if not os.path.isdir(folder):
            continue
        imgs = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        if len(imgs) >= 2:
            class_images[identity] = imgs

    identities = sorted(class_images.keys())
    paths1, paths2, pair_labels = [], [], []

    # Genuine pairs
    for identity in identities:
        imgs = class_images[identity]
        n_pairs = min(pairs_per_identity, len(imgs) * (len(imgs) - 1) // 2)
        for _ in range(n_pairs):
            i, j = rng.choice(len(imgs), size=2, replace=False)
            paths1.append(imgs[i])
            paths2.append(imgs[j])
            pair_labels.append(1)

    # Impostor pairs (same total count)
    n_genuine = len(pair_labels)
    for _ in range(n_genuine):
        id1, id2 = rng.choice(len(identities), size=2, replace=False)
        img1 = rng.choice(class_images[identities[id1]])
        img2 = rng.choice(class_images[identities[id2]])
        paths1.append(img1)
        paths2.append(img2)
        pair_labels.append(0)

    # Shuffle
    order = rng.permutation(len(paths1))
    paths1      = [paths1[i]      for i in order]
    paths2      = [paths2[i]      for i in order]
    pair_labels = [pair_labels[i] for i in order]

    print(f"[dataset] Verification pairs: {sum(pair_labels):,} genuine "
          f"+ {len(pair_labels) - sum(pair_labels):,} impostor "
          f"= {len(pair_labels):,} total.")
    return paths1, paths2, pair_labels
