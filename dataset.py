"""
VisiGuard Dataset Pipeline (ArcFace Ready + Stable VGGFace2 Loader)
====================================================================
Clean, non-duplicated, production-safe version.
"""

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
    logger.info(f"Dataset downloaded: {root}")
    return root


# ─────────────────────────────────────────────
# 2. Find VGGFace2 root
# ─────────────────────────────────────────────

def _find_vggface2_root(path: str) -> str:
    path = pathlib.Path(path)

    if (path / "train").exists():
        return str(path / "train")

    for p in path.rglob("*"):
        if p.is_dir():
            subdirs = list(p.iterdir())
            if len(subdirs) > 10:
                return str(p)

    return str(path)


# ─────────────────────────────────────────────
# 3. Scan dataset (clean + balanced)
# ─────────────────────────────────────────────

def scan_dataset(root: str):
    image_paths = []
    labels = []

    exts = {".jpg", ".jpeg", ".png"}

    identity_dirs = [d for d in pathlib.Path(root).iterdir() if d.is_dir()]

    logger.info(f"Found identities: {len(identity_dirs)}")

    kept = 0

    for identity_dir in identity_dirs:

        imgs = [
            str(p) for p in identity_dir.rglob("*")
            if p.suffix.lower() in exts
        ]

        if len(imgs) < config.MIN_IMAGES_PER_CLASS:
            continue

        if config.MAX_IMAGES_PER_CLASS:
            imgs = imgs[:config.MAX_IMAGES_PER_CLASS]

        image_paths.extend(imgs)
        labels.extend([identity_dir.name] * len(imgs))

        kept += 1
        if kept >= config.MAX_IDENTITIES:
            break

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    logger.info(
        f"Filtered → images={len(image_paths)} | identities={len(set(labels))}"
    )

    if len(set(labels)) < 2:
        raise ValueError("Too few identities after filtering")

    return image_paths, labels


# ─────────────────────────────────────────────
# 4. Encode + split
# ─────────────────────────────────────────────

def encode_and_split(paths, labels):

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X = np.array(paths)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=config.RANDOM_SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=0.5,
        stratify=y_tmp,
        random_state=config.RANDOM_SEED
    )

    logger.info(
        f"Split → train={len(X_train)} val={len(X_val)} test={len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, le, len(le.classes_)


# ─────────────────────────────────────────────
# 5. tf.data pipeline (ArcFace-friendly)
# ─────────────────────────────────────────────

def _load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


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

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def load_all():

    utils.ensure_dirs()

    root = download_dataset()
    root = _find_vggface2_root(root)

    logger.info(f"Using root: {root}")

    paths, labels = scan_dataset(root)

    X_train, X_val, X_test, y_train, y_val, y_test, le, num_classes = encode_and_split(
        paths, labels
    )

    utils.save_pickle(le, config.LABEL_ENCODER_PATH)

    train_ds = build_dataset(X_train, y_train, augment=True, shuffle=True)
    val_ds   = build_dataset(X_val, y_val)
    test_ds  = build_dataset(X_test, y_test)

    return train_ds, val_ds, test_ds, y_test, le, num_classes, labels