"""
VisiGuard Dataset Pipeline
==========================
Responsibilities:
  1. Download LFW-People from Kaggle via kagglehub
  2. Scan folder structure → build image-path / label lists
  3. Filter identities with too few images (class imbalance guard)
  4. Encode labels with sklearn LabelEncoder
  5. Stratified train / val / test split
  6. Build tf.data pipelines with:
     - resize + normalise (all splits)
     - augmentation (train only)
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
# 1.  Download dataset
# ─────────────────────────────────────────────

def download_dataset() -> str:
    """
    Download the LFW-People dataset via kagglehub.
    Returns the local root path containing identity sub-folders.
    """
    import kagglehub
    logger.info(f"Downloading dataset: {config.KAGGLE_DATASET} …")
    root = kagglehub.dataset_download(config.KAGGLE_DATASET)
    logger.info(f"Dataset root: {root}")

    # kagglehub may nest one extra folder; find the folder with sub-dirs
    root = _find_image_root(root)
    logger.info(f"Image root detected: {root}")
    return root


def _find_image_root(path: str) -> str:
    """
    Walk down until we find a directory whose children are all directories
    (i.e., the identity-level folder, not a nested archive folder).
    """
    for dirpath, dirnames, filenames in os.walk(path):
        # If this folder contains sub-folders with images → it's our root
        if dirnames:
            # Verify at least one sub-folder has images inside
            sample = os.path.join(dirpath, dirnames[0])
            imgs = glob.glob(os.path.join(sample, "*.jpg")) + \
                   glob.glob(os.path.join(sample, "*.png")) + \
                   glob.glob(os.path.join(sample, "*.jpeg"))
            if imgs:
                return dirpath
    return path    # fallback: use given path as-is


# ─────────────────────────────────────────────
# 2.  Scan & filter
# ─────────────────────────────────────────────

def scan_dataset(root: str) -> tuple[list[str], list[str]]:
    """
    Walk the identity sub-folders and collect (image_path, label) pairs.
    Filters out any identity with fewer than MIN_IMAGES_PER_CLASS images.

    Returns
    -------
    image_paths : list of absolute file paths
    labels      : list of identity strings (one per path)
    """
    image_paths: list[str] = []
    labels:      list[str] = []

    extensions = {".jpg", ".jpeg", ".png"}

    identity_dirs = sorted([
        d for d in pathlib.Path(root).iterdir() if d.is_dir()
    ])
    logger.info(f"Total identities found in dataset: {len(identity_dirs)}")

    for identity_dir in identity_dirs:
        imgs = [
            str(p) for p in identity_dir.iterdir()
            if p.suffix.lower() in extensions
        ]
        if len(imgs) < config.MIN_IMAGES_PER_CLASS:
            continue   # skip rare identities
        image_paths.extend(imgs)
        labels.extend([identity_dir.name] * len(imgs))

    logger.info(
        f"After filtering (≥{config.MIN_IMAGES_PER_CLASS} imgs): "
        f"{len(set(labels))} identities, {len(image_paths)} images total."
    )

    if len(set(labels)) < 2:
        raise ValueError(
            "Too few classes survived filtering. "
            "Lower MIN_IMAGES_PER_CLASS in config.py."
        )

    return image_paths, labels


# ─────────────────────────────────────────────
# 3.  Label encoding + splitting
# ─────────────────────────────────────────────

def encode_and_split(
    image_paths: list[str],
    labels:      list[str],
) -> tuple:
    """
    Encode string labels → integer class indices, then perform a
    stratified Train / Val / Test split.

    Returns
    -------
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     label_encoder, num_classes)
    """
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    num_classes = len(le.classes_)
    logger.info(f"Number of classes: {num_classes}")

    X = np.array(image_paths)

    # First cut: train vs (val + test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=(1.0 - config.TRAIN_RATIO),
        stratify=y,
        random_state=config.RANDOM_SEED,
    )

    # Second cut: val vs test from the remainder
    relative_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_test,
        stratify=y_tmp,
        random_state=config.RANDOM_SEED,
    )

    logger.info(
        f"Split sizes → train: {len(X_train)}, "
        f"val: {len(X_val)}, test: {len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, le, num_classes


# ─────────────────────────────────────────────
# 4.  tf.data pipeline helpers
# ─────────────────────────────────────────────

def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor):
    """
    Read → decode → resize → normalise a single image.
    Called inside tf.data map(); runs on CPU in graph mode.
    """
    raw   = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)          # handles PNG too
    image = tf.image.resize(image, config.IMAGE_SIZE)      # (H, W)
    # EfficientNet expects pixels in [0, 255]; its own rescaling layer will
    # handle normalisation internally.  We cast to float32 here.
    image = tf.cast(image, tf.float32)
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor):
    """
    Apply stochastic augmentation for training robustness.
    All ops are differentiable-safe and run inside tf.data pipeline.

    Augmentations applied:
      • Random horizontal flip
      • Random rotation  (±15°)
      • Random brightness adjustment
      • Random contrast  adjustment
      • Random zoom-crop (±10 %)
    """
    # Horizontal flip
    if config.AUG_HFLIP:
        image = tf.image.random_flip_left_right(image)

    # Brightness
    image = tf.image.random_brightness(
        image, max_delta=config.AUG_BRIGHTNESS_DELTA
    )

    # Contrast
    image = tf.image.random_contrast(
        image,
        lower=config.AUG_CONTRAST_LOWER,
        upper=config.AUG_CONTRAST_UPPER,
    )

    # Rotation via tfa-free method: random crop + pad + crop to original size
    # We achieve zoom-like augmentation using random crop
    h, w, c = config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3
    zoom_frac = config.AUG_ZOOM_RANGE
    crop_h = int(h * (1.0 - zoom_frac))
    crop_w = int(w * (1.0 - zoom_frac))
    image = tf.image.random_crop(image, size=[crop_h, crop_w, c])
    image = tf.image.resize(image, config.IMAGE_SIZE)

    # Clip to valid float range for EfficientNet [0, 255]
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image, label


def build_dataset(
    paths:     np.ndarray,
    labels:    np.ndarray,
    augment:   bool = False,
    shuffle:   bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from file-path + integer-label arrays.

    Parameters
    ----------
    paths   : array of image file-path strings
    labels  : array of integer class indices
    augment : if True, apply random augmentation (training split only)
    shuffle : if True, shuffle before batching (training split only)

    Returns
    -------
    Batched, prefetched tf.data.Dataset ready for model.fit()
    """
    ds = tf.data.Dataset.from_tensor_slices(
        (paths.astype(str), labels.astype(np.int32))
    )

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 5000),
            seed=config.RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(_load_and_preprocess,
                num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = (
        ds
        .batch(config.BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds


# ─────────────────────────────────────────────
# 5.  Public convenience function
# ─────────────────────────────────────────────

def load_all() -> tuple:
    """
    Full pipeline in one call:
      download → scan → filter → encode → split → build datasets

    Returns
    -------
    train_ds, val_ds, test_ds  : tf.data.Dataset (ready for training)
    y_test                     : numpy array (raw integer labels for eval)
    label_encoder              : fitted sklearn LabelEncoder
    num_classes                : int
    all_labels                 : all raw string labels (for distribution plot)
    """
    utils.ensure_dirs()
    root        = download_dataset()
    paths, lbls = scan_dataset(root)

    # Plot class distribution before filtering
    utils.plot_class_distribution(lbls, title="Identity Sample Distribution (filtered)")

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     le, num_classes) = encode_and_split(paths, lbls)

    # Save label encoder immediately
    utils.save_pickle(le, config.LABEL_ENCODER_PATH)

    # Build tf.data pipelines
    train_ds = build_dataset(X_train, y_train, augment=True,  shuffle=True)
    val_ds   = build_dataset(X_val,   y_val,   augment=False, shuffle=False)
    test_ds  = build_dataset(X_test,  y_test,  augment=False, shuffle=False)

    return train_ds, val_ds, test_ds, y_test, le, num_classes, lbls
