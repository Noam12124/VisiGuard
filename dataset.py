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

    logger.info(f"Downloading dataset: {config.KAGGLE_DATASET} …")
    root = kagglehub.dataset_download(config.KAGGLE_DATASET)
    logger.info(f"Dataset root: {root}")

    return root


# ─────────────────────────────────────────────
# 2. Find correct VGGFace2 root
# ─────────────────────────────────────────────

def _find_vggface2_root(path: str) -> str:
    """
    VGGFace2 structure:
        root/
            train/
                n000001/
                n000002/
            test/
    We prefer TRAIN split for training.
    """
    path = pathlib.Path(path)

    # Try direct VGGFace2 structure
    train_path = path / "train"
    if train_path.exists():
        return str(train_path)

    # fallback: search for folder containing identity folders
    for p in path.rglob("*"):
        if p.is_dir():
            subdirs = list(p.iterdir())
            if len(subdirs) > 10:
                return str(p)

    return str(path)


# ─────────────────────────────────────────────
# 3. Scan dataset (FIXED FOR VGGFACE2)
# ─────────────────────────────────────────────

def scan_dataset(root: str) -> tuple[list[str], list[str]]:

    image_paths = []
    labels = []

    extensions = {".jpg", ".jpeg", ".png"}

    identity_dirs = [
        d for d in pathlib.Path(root).iterdir()
        if d.is_dir()
    ]

    logger.info(f"Raw identities found: {len(identity_dirs)}")

    # ── ADAPTIVE FILTER (IMPORTANT FOR 85% TARGET)
    # VGGFace2 is large → do NOT over-filter
    min_images = min(config.MIN_IMAGES_PER_CLASS, 5)

    logger.info(f"Using MIN_IMAGES_PER_CLASS = {min_images}")

    kept = 0

    for identity_dir in identity_dirs:
        imgs = [
            str(p) for p in identity_dir.rglob("*")
            if p.suffix.lower() in extensions
        ]

        if len(imgs) < min_images:
            continue

        image_paths.extend(imgs)
        labels.extend([identity_dir.name] * len(imgs))
        kept += 1

        # safety cap (prevents RAM explosion)
        if kept >= 5000:
            break

    logger.info(
        f"Filtered → {len(set(labels))} identities | "
        f"{len(image_paths)} images"
    )

    if len(set(labels)) < 2:
        raise ValueError("Too few classes after filtering — reduce MIN_IMAGES_PER_CLASS")

    return image_paths, labels


# ─────────────────────────────────────────────
# 4. Encode + split (IMPROVED BALANCE)
# ─────────────────────────────────────────────

def encode_and_split(image_paths, labels):

    le = LabelEncoder()
    y = le.fit_transform(labels)

    num_classes = len(le.classes_)
    logger.info(f"Classes: {num_classes}")

    X = np.array(image_paths)

    # stratified split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=0.5,
        stratify=y_tmp,
        random_state=config.RANDOM_SEED,
    )

    logger.info(
        f"Split → train={len(X_train)} val={len(X_val)} test={len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, le, num_classes


# ─────────────────────────────────────────────
# 5. tf.data pipeline (same but stable)
# ─────────────────────────────────────────────

def _load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def build_dataset(paths, labels, augment=False, shuffle=False):

    ds = tf.data.Dataset.from_tensor_slices(
        (paths.astype(str), labels.astype(np.int32))
    )

    if shuffle:
        ds = ds.shuffle(min(len(paths), 5000))

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def load_all():

    utils.ensure_dirs()

    root = download_dataset()

    # FIX: proper VGGFace2 root detection
    root = _find_vggface2_root(root)

    logger.info(f"Using dataset root: {root}")

    paths, lbls = scan_dataset(root)

    utils.plot_class_distribution(lbls, title="VGGFace2 Distribution")

    X_train, X_val, X_test, y_train, y_val, y_test, le, num_classes = encode_and_split(
        paths, lbls
    )

    utils.save_pickle(le, config.LABEL_ENCODER_PATH)

    train_ds = build_dataset(X_train, y_train, augment=True, shuffle=True)
    val_ds   = build_dataset(X_val, y_val, augment=False)
    test_ds  = build_dataset(X_test, y_test, augment=False)

    return train_ds, val_ds, test_ds, y_test, le, num_classes, lbls