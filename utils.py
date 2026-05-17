"""
VisiGuard Dataset Pipeline (VGGFace2 Optimized - FIXED VERSION)
================================================================
Stable, production-safe loader for face recognition training.
"""

import os
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
    Finds identity root folder in VGGFace2 structure.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        if dirnames:
            sample_dir = os.path.join(dirpath, dirnames[0])
            if len(list(pathlib.Path(sample_dir).glob("*.jpg"))) > 5:
                return dirpath
    return path


# ─────────────────────────────────────────────
# 2. Scan dataset (FIXED + balanced)
# ─────────────────────────────────────────────

def scan_dataset(root: str):
    image_paths = []
    labels = []

    extensions = {".jpg", ".jpeg", ".png"}

    identity_dirs = [
        d for d in pathlib.Path(root).rglob("*")
        if d.is_dir()
    ]

    logger.info(f"Raw folders found: {len(identity_dirs)}")

    for identity_dir in identity_dirs:

        imgs = [
            p for p in identity_dir.glob("*")
            if p.suffix.lower() in extensions
        ]

        if len(imgs) < config.MIN_IMAGES_PER_CLASS:
            continue

        # ✔ FIX: random sampling instead of slicing
        if hasattr(config, "MAX_IMAGES_PER_CLASS") and config.MAX_IMAGES_PER_CLASS:
            imgs = np.random.choice(
                imgs,
                size=min(len(imgs), config.MAX_IMAGES_PER_CLASS),
                replace=False
            ).tolist()

        image_paths.extend([str(p) for p in imgs])
        labels.extend([identity_dir.name] * len(imgs))

    logger.info(
        f"After filtering → Classes: {len(set(labels))} | Images: {len(image_paths)}"
    )

    if len(set(labels)) < 2:
        raise ValueError("Too few classes after filtering dataset!")

    return np.array(image_paths), np.array(labels)


# ─────────────────────────────────────────────
# 3. Encode + identity-safe split (FIXED)
# ─────────────────────────────────────────────

def encode_and_split(image_paths, labels):

    le = LabelEncoder()
    y = le.fit_transform(labels)

    logger.info(f"Total classes: {len(le.classes_)}")

    # ✔ FIX: identity-safe split (NO leakage)
    unique_classes = np.unique(y)

    train_ids, temp_ids = train_test_split(
        unique_classes,
        test_size=1 - config.TRAIN_RATIO,
        random_state=config.RANDOM_SEED
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=config.RANDOM_SEED
    )

    def build_subset(ids):
        mask = np.isin(y, ids)
        return image_paths[mask], y[mask]

    X_train, y_train = build_subset(train_ids)
    X_val, y_val = build_subset(val_ids)
    X_test, y_test = build_subset(test_ids)

    logger.info(
        f"Split → train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, le, len(le.classes_)


# ─────────────────────────────────────────────
# 4. Dataset pipeline (FIXED decode + safe crop)
# ─────────────────────────────────────────────

def _load_and_preprocess(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)  # FIXED (faster + stable)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


def _augment(image, label):

    if config.AUG_HFLIP:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, config.AUG_BRIGHTNESS_DELTA)
    image = tf.image.random_contrast(
        image,
        config.AUG_CONTRAST_LOWER,
        config.AUG_CONTRAST_UPPER
    )

    # FIXED: safer face augmentation (no aggressive cropping)
    image = tf.image.central_crop(
        image,
        central_fraction=1.0 - config.AUG_ZOOM_RANGE
    )
    image = tf.image.resize(image, config.IMAGE_SIZE)

    return tf.clip_by_value(image, 0.0, 255.0), label


def build_dataset(paths, labels, augment=False, shuffle=False):

    ds = tf.data.Dataset.from_tensor_slices((paths, labels.astype(np.int32)))

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