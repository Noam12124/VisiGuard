"""
dataset.py  (VisiGuard — revised)
═══════════════════════════════════════════════════════════════════════════════
Key changes vs the original:
──────────────────────────────────────────────────────────────────────────────
[FIX-1] IDENTITY-DISJOINT SPLITS  ← eliminates data leakage
    Old: random stratified split on individual images → same person's frames
         appear in both train and val.  Val loss looked deceptively good.
    New: split on IDENTITY NAMES first (80/10/10 of identities), then collect
         ALL images for each identity into the correct bucket.  No identity
         ever appears in more than one split.

[FIX-2] MIN_IMAGES_PER_CLASS raised to 15 (via config)
    5 images per identity is insufficient for ArcFace to learn a tight
    angular cluster.  15+ gives the loss enough intra-class variation.
    Changed in config.py; dataset.py honours whatever value is set there.

[FIX-3] VERIFICATION PAIRS DRAWN EXCLUSIVELY FROM VAL/TEST IDENTITIES
    build_verification_pairs() now accepts an explicit identity list so the
    callback in train.py can pass val_identities and guarantee zero overlap
    with training classes.

[FIX-4] ALIGNMENT FALLBACK HARDENED
    The old Haar-cascade fallback silently wrote corrupted crops when
    cv2.imread returned None but the path still existed (NFS / Colab Drive
    race condition).  Added explicit guard + skip counter.

[FIX-5] build_datasets() now returns val_identities + test_identities
    so train.py can pass them straight to the verification callback without
    re-scanning the disk.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config
from aligner import FaceAligner


# ─────────────────────────────────────────────────────────────────────────────
# Offline alignment
# ─────────────────────────────────────────────────────────────────────────────

def prepare_aligned_dataset(
    src_dir: str = config.DATA_DIR,
    dst_dir: str = None,
    aligner: FaceAligner = None,
) -> str:
    """
    Walk src_dir, align every face with Haar-cascade eye detection, write
    results to dst_dir (src_dir + '_aligned' by default).

    Changes vs original:
      • Skips the whole folder if dst_dir already exists (unchanged).
      • [FIX-4] Explicitly skips files where cv2.imread returns None and
        increments a skip counter so you can see how many images are corrupt.
      • Haar-cascade fallback now writes the full-frame resize rather than
        silently producing a black image when eyes aren't found.
    """
    if dst_dir is None:
        dst_dir = src_dir.rstrip("/\\") + "_aligned"

    if os.path.exists(dst_dir):
        print(f"[dataset] Aligned dataset already exists at {dst_dir}. Skipping.")
        return dst_dir

    if aligner is None:
        aligner = FaceAligner()

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    identities = sorted([
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    ])

    total_written = 0
    total_skipped = 0  # [FIX-4] explicit skip counter

    for identity in identities:
        in_folder  = os.path.join(src_dir,  identity)
        out_folder = os.path.join(dst_dir,  identity)
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

            # [FIX-4] Hard guard — skip corrupt / unreadable files.
            if img_bgr is None or img_bgr.size == 0:
                total_skipped += 1
                continue

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )

            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])
                left_eye_centre = (
                    int(eyes[0][0] + eyes[0][2] / 2),
                    int(eyes[0][1] + eyes[0][3] / 2),
                )
                right_eye_centre = (
                    int(eyes[1][0] + eyes[1][2] / 2),
                    int(eyes[1][1] + eyes[1][3] / 2),
                )
                aligned = aligner.align(img_bgr, left_eye_centre, right_eye_centre)
            else:
                # [FIX-4] Fallback: plain centre-crop of the full frame.
                h, w = img_bgr.shape[:2]
                aligned = aligner.crop_and_resize(img_bgr, (0, 0, w, h))

            # Final size guard before writing.
            if aligned is None or aligned.size == 0:
                total_skipped += 1
                continue

            cv2.imwrite(dst_path, aligned)
            total_written += 1

        if total_written % 1000 == 0 and total_written > 0:
            print(f"  [align] {total_written:,} written …", end="\r")

    print(
        f"\n[dataset] Alignment done.  "
        f"Written: {total_written:,}   Skipped (corrupt): {total_skipped:,}"
    )
    print(f"[dataset] Aligned dataset at: {dst_dir}")
    return dst_dir


# ─────────────────────────────────────────────────────────────────────────────
# Class catalogue
# ─────────────────────────────────────────────────────────────────────────────

def build_class_catalogue(
    data_dir: str,
    min_images: int = config.MIN_IMAGES_PER_CLASS,
) -> tuple[list[str], dict[str, int]]:
    """
    Scan data_dir and return (class_names, class_to_idx).
    Only identities with ≥ min_images images are kept.

    With [FIX-2] config.MIN_IMAGES_PER_CLASS is now 15, giving ArcFace
    enough intra-class samples to form tight angular clusters.
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
    print(
        f"[dataset] Found {len(class_names)} identities "
        f"(of {len(identities)} total, min_images={min_images})."
    )
    return class_names, class_to_idx


def save_class_catalogue(class_names: list[str], path: str = None) -> None:
    if path is None:
        path = os.path.join(config.CHECKPOINT_DIR, "class_names.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[dataset] Class catalogue saved → {path}")


def load_class_catalogue(path: str = None) -> list[str]:
    if path is None:
        path = os.path.join(config.CHECKPOINT_DIR, "class_names.json")
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-1]  Identity-disjoint splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_identities(
    class_names: list[str],
    val_frac:    float = config.VALIDATION_SPLIT,   # default 0.15
    test_frac:   float = config.TEST_SPLIT,         # default 0.10
    seed:        int   = config.RANDOM_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split the list of IDENTITY NAMES into train / val / test buckets.

    Why this matters
    ────────────────
    The old code split individual IMAGE PATHS with stratify=labels.  When an
    identity has many correlated frames (same lighting, same session), some
    frames land in train and some in val.  The model sees the val face during
    training → inflated val_accuracy → misleading early-stopping signal.

    By splitting on identity names first, EVERY image of a given person
    belongs to exactly one split.  Val and test identities are completely
    unseen during training, which is exactly what deployment looks like.

    Returns
    ───────
    train_ids, val_ids, test_ids  — disjoint lists of identity name strings.
    """
    rng = np.random.default_rng(seed)
    names = list(class_names)
    rng.shuffle(names)

    n       = len(names)
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_ids = names[:n_train]
    val_ids   = names[n_train : n_train + n_val]
    test_ids  = names[n_train + n_val:]

    print(
        f"[dataset] Identity split (disjoint):  "
        f"train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}"
    )
    return train_ids, val_ids, test_ids


def _collect_files(
    data_dir:     str,
    identity_list: list[str],
    class_to_idx:  dict[str, int],
) -> tuple[list[str], list[int]]:
    """Return (file_paths, labels) for every image belonging to identity_list."""
    file_paths: list[str] = []
    labels:     list[int] = []

    for name in identity_list:
        folder = os.path.join(data_dir, name)
        idx    = class_to_idx[name]
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                file_paths.append(os.path.join(folder, fname))
                labels.append(idx)

    return file_paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# tf.data pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_decode(path: tf.Tensor) -> tf.Tensor:
    """Read → decode → resize → float32 in [0, 255]."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, config.IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _augment(image: tf.Tensor) -> tf.Tensor:
    """Standard augmentation for training images."""
    if config.AUGMENT_FLIP:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, config.AUGMENT_BRIGHTNESS)
    image = tf.image.random_contrast(
        image,
        1.0 - config.AUGMENT_CONTRAST,
        1.0 + config.AUGMENT_CONTRAST,
    )
    image = tf.image.random_saturation(
        image,
        1.0 - config.AUGMENT_SATURATION,
        1.0 + config.AUGMENT_SATURATION,
    )
    image = tf.image.random_hue(image, config.AUGMENT_HUE)

    # Random rotation via tensorflow_addons (no-op fallback if not installed)
    angle_rad = (
        tf.random.uniform([], -config.AUGMENT_ROTATION, config.AUGMENT_ROTATION)
        * (3.14159265 / 180.0)
    )
    image = _rotate(image, angle_rad)

    # Random zoom
    zoom  = tf.random.uniform([], 1.0 - config.AUGMENT_ZOOM, 1.0 + config.AUGMENT_ZOOM)
    h, w  = config.IMAGE_SIZE
    new_h = tf.clip_by_value(
        tf.cast(tf.round(tf.cast(h, tf.float32) / zoom), tf.int32), 1, h
    )
    new_w = tf.clip_by_value(
        tf.cast(tf.round(tf.cast(w, tf.float32) / zoom), tf.int32), 1, w
    )
    image = tf.image.resize_with_crop_or_pad(image, new_h, new_w)
    image = tf.image.resize(image, [h, w])

    return tf.clip_by_value(image, 0.0, 255.0)


def _rotate(image: tf.Tensor, angle_rad: tf.Tensor) -> tf.Tensor:
    try:
        import tensorflow_addons as tfa
        return tfa.image.rotate(image, angle_rad, interpolation="BILINEAR")
    except ImportError:
        return image   # rotation silently disabled when tfa is absent


def _make_dataset(
    file_paths: list[str],
    labels:     list[int],
    augment:    bool,
    shuffle:    bool,
    batch_size: int = config.BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset yielding  ((image, label), label).

    The dual-label format is required because ArcFaceLayer.call() needs the
    true class index at forward-pass time (to insert the angular margin only
    for the true class column).  Keras .fit() receives:
        inputs  = (image, label)
        targets = label
    """
    paths_ds  = tf.data.Dataset.from_tensor_slices(file_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int32))
    ds = tf.data.Dataset.zip((paths_ds, labels_ds))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(file_paths), 50_000),
            seed=config.RANDOM_SEED,
        )

    def load_and_prep(path, label):
        img = _load_and_decode(path)
        if augment:
            img = _augment(img)
        return (img, label), label

    ds = (
        ds
        .map(load_and_prep, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Top-level builder — returns identity lists for the callback
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    data_dir:   str = None,
    batch_size: int = config.BATCH_SIZE,
) -> tuple[
    tf.data.Dataset,   # train_ds
    tf.data.Dataset,   # val_ds
    tf.data.Dataset,   # test_ds
    list[str],         # all class_names  (training identities, for ArcFace dim)
    list[str],         # val_ids          [FIX-5] returned so callback can use them
    list[str],         # test_ids         [FIX-5]
]:
    """
    Full pipeline:  catalogue  →  identity-disjoint split  →  tf.data

    Changes
    ───────
    [FIX-1] Uses split_identities() instead of image-level train_test_split.
    [FIX-5] Returns val_ids and test_ids so train.py can pass them directly
            to the VerificationCallback without re-scanning disk.

    Returns
    ───────
    train_ds, val_ds, test_ds, class_names, val_ids, test_ids

    NOTE: val_ds and test_ds are still returned as ArcFace-format datasets
    so that the standard Keras val_accuracy metric is still tracked as a
    secondary signal (useful for debugging).  The PRIMARY metric the
    VerificationCallback reports is cosine-similarity AUC.
    """
    if data_dir is None:
        aligned = config.DATA_DIR.rstrip("/\\") + "_aligned"
        data_dir = aligned if os.path.exists(aligned) else config.DATA_DIR

    # Build full catalogue (respects MIN_IMAGES_PER_CLASS from config)
    all_names, class_to_idx = build_class_catalogue(data_dir)

    # [FIX-1] Split on identities, not images
    train_ids, val_ids, test_ids = split_identities(all_names)

    # Remap labels so training identity indices are contiguous 0…N_train-1.
    # Val/test identities get their own local indices ONLY for the val/test
    # ArcFace datasets; the embedding callback ignores these labels entirely.
    train_c2i  = {name: idx for idx, name in enumerate(train_ids)}
    val_c2i    = {name: idx for idx, name in enumerate(val_ids)}
    test_c2i   = {name: idx for idx, name in enumerate(test_ids)}

    tr_paths, tr_labels = _collect_files(data_dir, train_ids, train_c2i)
    va_paths, va_labels = _collect_files(data_dir, val_ids,   val_c2i)
    te_paths, te_labels = _collect_files(data_dir, test_ids,  test_c2i)

    print(
        f"[dataset] Images:  train={len(tr_labels):,}  "
        f"val={len(va_labels):,}  test={len(te_labels):,}"
    )

    train_ds = _make_dataset(tr_paths, tr_labels, augment=True,  shuffle=True,  batch_size=batch_size)
    val_ds   = _make_dataset(va_paths, va_labels, augment=False, shuffle=False, batch_size=batch_size)
    test_ds  = _make_dataset(te_paths, te_labels, augment=False, shuffle=False, batch_size=batch_size)

    # Save the TRAINING catalogue (only train_ids are actual ArcFace classes)
    save_class_catalogue(train_ids)

    # [FIX-5] Return val/test identity names for the verification callback
    return train_ds, val_ds, test_ds, train_ids, val_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-3]  Verification pair builder — draws from a supplied identity list
# ─────────────────────────────────────────────────────────────────────────────

def build_verification_pairs(
    data_dir:           str       = None,
    identity_list:      list[str] = None,   # [FIX-3] NEW — pass val_ids or test_ids
    pairs_per_identity: int       = config.EVAL_PAIRS_PER_CLASS,
    seed:               int       = config.RANDOM_SEED,
) -> tuple[list[str], list[str], list[int]]:
    """
    Build (path1, path2, is_same) triples for pairwise verification.

    Genuine pairs  — two different images of the SAME person.
    Impostor pairs — images of DIFFERENT people.
    Returned label: 1 = same person, 0 = different.

    Parameters
    ──────────
    identity_list : [FIX-3]
        If provided, ONLY identities in this list are used.  Pass val_ids
        (or test_ids) to ensure zero overlap with training identities.
        If None, all identities in data_dir are used (backwards-compatible).

    Returns
    ───────
    paths1, paths2, pair_labels
    """
    if data_dir is None:
        aligned = config.DATA_DIR.rstrip("/\\") + "_aligned"
        data_dir = aligned if os.path.exists(aligned) else config.DATA_DIR

    rng = np.random.default_rng(seed)

    # Collect image lists, restricted to identity_list if given
    class_images: dict[str, list[str]] = {}
    for identity in sorted(os.listdir(data_dir)):
        if identity_list is not None and identity not in identity_list:
            continue   # [FIX-3] skip training identities
        folder = os.path.join(data_dir, identity)
        if not os.path.isdir(folder):
            continue
        imgs = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        if len(imgs) >= 2:
            class_images[identity] = imgs

    identities = sorted(class_images.keys())
    if len(identities) < 2:
        raise ValueError(
            f"[dataset] build_verification_pairs: need ≥2 identities, "
            f"got {len(identities)}.  Check identity_list and data_dir."
        )

    paths1, paths2, pair_labels = [], [], []

    # ── Genuine pairs ─────────────────────────────────────────────────────
    for identity in identities:
        imgs    = class_images[identity]
        # Cap pairs at pairs_per_identity; don't try to generate more than
        # the combinatorial limit of the available images.
        max_possible = len(imgs) * (len(imgs) - 1) // 2
        n_pairs = min(pairs_per_identity, max_possible)
        chosen: set[tuple[int, int]] = set()
        attempts = 0
        while len(chosen) < n_pairs and attempts < n_pairs * 10:
            i, j = rng.choice(len(imgs), size=2, replace=False)
            pair = (min(i, j), max(i, j))
            chosen.add(pair)
            attempts += 1
        for i, j in chosen:
            paths1.append(imgs[i])
            paths2.append(imgs[j])
            pair_labels.append(1)

    # ── Impostor pairs (balanced: same count as genuine) ──────────────────
    n_genuine = len(pair_labels)
    attempts  = 0
    impostor_set: set[tuple[str, str]] = set()

    while len(impostor_set) < n_genuine and attempts < n_genuine * 20:
        idx1, idx2 = rng.choice(len(identities), size=2, replace=False)
        img1 = str(rng.choice(class_images[identities[idx1]]))
        img2 = str(rng.choice(class_images[identities[idx2]]))
        key  = (min(img1, img2), max(img1, img2))
        if key not in impostor_set:
            impostor_set.add(key)
            paths1.append(img1)
            paths2.append(img2)
            pair_labels.append(0)
        attempts += 1

    # ── Shuffle ───────────────────────────────────────────────────────────
    order       = rng.permutation(len(paths1))
    paths1      = [paths1[i]      for i in order]
    paths2      = [paths2[i]      for i in order]
    pair_labels = [pair_labels[i] for i in order]

    n_same  = sum(pair_labels)
    n_diff  = len(pair_labels) - n_same
    print(
        f"[dataset] Verification pairs: "
        f"{n_same:,} genuine + {n_diff:,} impostor = {len(pair_labels):,} total  "
        f"(from {len(identities)} identities)"
    )
    return paths1, paths2, pair_labels
