"""
VisiGuard Dataset Pipeline (VGGFace2 Optimized)
===============================================
Drop-in replacement tuned for VGGFace2 structure.

Key improvements:
- Fully recursive dataset scan (VGGFace2 compatible)
- Identity balancing (cap images per class)
- Better filtering strategy (prevents dominance of huge identities)
- Faster glob-based loading
"""

import os
import glob
import pathlib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import config
import utils

logger = utils.get_logger()


# ─────────────────────────────────────────────
# 1. Download dataset
# ─────────────────────────────────────────────

def download_dataset() -> str:
    import kagglehub

    logger.info(f"Downloading dataset: {config.KAGGLE_DATASET}")
    root = kagglehub.dataset_download(config.KAGGLE_DATASET)

    root = _find_image_root(root)

    logger.info(f"Dataset root resolved: {root}")
    return root


def _find_image_root(path: str) -> str:
    """
    Finds the actual identity root in VGGFace2 (handles train/test splits).
    """
    # VGGFace2 often has train/ folder
    for dirpath, dirnames, filenames in os.walk(path):
        if dirnames:
            sample_dir = os.path.join(dirpath, dirnames[0])

            imgs = glob.glob(os.path.join(sample_dir, "*.jpg"))
            if len(imgs) > 5:
                return dirpath

    return path


# ─────────────────────────────────────────────
# 2. Scan dataset (VGGFace2 optimized)
# ─────────────────────────────────────────────

def scan_dataset(root: str):
    """
    Recursively scans VGGFace2 identity folders.
    Adds optional per-class cap for balance.
    """

    image_paths = []
    labels = []

    extensions = {".jpg", ".jpeg", ".png"}

    identity_dirs = [
        d for d in pathlib.Path(root).rglob("*")
        if d.is_dir() and len(list(d.glob("*"))) > 0
    ]

    logger.info(f"Raw folders found: {len(identity_dirs)}")

    for identity_dir in identity_dirs:

        imgs = [
            str(p) for p in identity_dir.glob("*")
            if p.suffix.lower() in extensions
        ]

        if len(imgs) < config.MIN_IMAGES_PER_CLASS:
            continue

        # ── IMPORTANT: balance huge VGGFace2 classes ──
        if hasattr(config, "MAX_IMAGES_PER_CLASS") and config.MAX_IMAGES_PER_CLASS:
            imgs = imgs[:config.MAX_IMAGES_PER_CLASS]

        image_paths.extend(imgs)
        labels.extend([identity_dir.name] * len(imgs))

    logger.info(
        f"After filtering → "
        f"Classes: {len(set(labels))} | Images: {len(image_paths)}"
    )

    if len(set(labels)) < 2:
        raise ValueError("Too few classes after filtering.")

    return image_paths, labels


# ─────────────────────────────────────────────
# 3. Encode + split (FIXED for large datasets)
# ─────────────────────────────────────────────

def encode_and_split(image_paths, labels):

    le = LabelEncoder()
    y = le.fit_transform(labels)

    num_classes = len(le.classes_)
    logger.info(f"Total classes: {num_classes}")

    X = np.array(image_paths)

    # Stratified split (safe for large class count)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=1 - config.TRAIN_RATIO,
        stratify=y,
        random_state=config.RANDOM_SEED
    )

    relative_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_test,
        stratify=y_tmp,
        random_state=config.RANDOM_SEED
    )

    logger.info(
        f"Split → train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, le, num_classes


# ─────────────────────────────────────────────
# 4. tf.data pipeline (unchanged but stable)
# ─────────────────────────────────────────────

def _load_and_preprocess(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


def _augment(image, label):
    if config.AUG_HFLIP:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(
        image, config.AUG_BRIGHTNESS_DELTA
    )

    image = tf.image.random_contrast(
        image,
        config.AUG_CONTRAST_LOWER,
        config.AUG_CONTRAST_UPPER
    )

    h, w = config.IMAGE_SIZE
    crop_h = int(h * (1 - config.AUG_ZOOM_RANGE))
    crop_w = int(w * (1 - config.AUG_ZOOM_RANGE))

    image = tf.image.random_crop(image, [crop_h, crop_w, 3])
    image = tf.image.resize(image, config.IMAGE_SIZE)

    return tf.clip_by_value(image, 0.0, 255.0), label


def build_dataset(paths, labels, augment=False, shuffle=False):

    ds = tf.data.Dataset.from_tensor_slices(
        (paths.astype(str), labels.astype(np.int32))
    )

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 10000),
            seed=config.RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────
# 5. Main loader
# ─────────────────────────────────────────────

def load_all():

    utils.ensure_dirs()

    root = download_dataset()
    paths, labels = scan_dataset(root)

    utils.plot_class_distribution(labels, "VGGFace2 Distribution")

    X_train, X_val, X_test, y_train, y_val, y_test, le, num_classes = \
        encode_and_split(paths, labels)

    utils.save_pickle(le, config.LABEL_ENCODER_PATH)

    train_ds = build_dataset(X_train, y_train, augment=True, shuffle=True)
    val_ds   = build_dataset(X_val, y_val)
    test_ds  = build_dataset(X_test, y_test)

    return train_ds, val_ds, test_ds, y_test, le, num_classes, labels