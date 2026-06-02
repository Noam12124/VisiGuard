"""
dataset.py  (VisiGuard — revised)
═══════════════════════════════════════════════════════════════════════════════
Key fixes in this version:
──────────────────────────────────────────────────────────────────────────────
[FIX-1] IDENTITY-DISJOINT SPLITS — no data leakage
    Splits on IDENTITY NAMES first (80/10/10), then assigns all of each
    person's images to the correct bucket.  No identity spans two splits.

[FIX-2] MIN_IMAGES_PER_CLASS = 5 (in config)
    Keeps all ~1,680 LFW identities instead of filtering down to ~158.
    More identities = more angular decision boundaries for ArcFace.

[FIX-3] VERIFICATION PAIRS FROM VAL/TEST IDENTITIES ONLY
    build_verification_pairs() accepts an explicit identity list so the
    callback guarantees zero overlap with training classes.

[FIX-4] ALIGNMENT FALLBACK HARDENED
    Guards against race-condition None reads and 0-byte files.

[FIX-5] build_datasets() return order corrected
    Now returns: train_ds, val_ds, test_ds, train_identities,
                 val_identities, test_identities
    (Previously val_identities was in position 4 and num_classes in
    position 6, which caused train.py to build the model with the wrong
    number of classes and pass an integer as identity list.)

[FIX-6] Added missing build_class_catalogue() and prepare_aligned_dataset()
    Referenced from training_demo.ipynb but absent from the original file.
"""

import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import config


# ── Custom Augmentation Layer — Random Occlusion Erasing ──────────────────

class RandomOcclusionErasing(layers.Layer):
    """
    Simulates real-world partial occlusions (sunglasses, hands, masks) by
    randomly overwriting a rectangular patch with uniform noise.

    Compatible with mixed-precision (float16) training and tf.data Graph Mode.
    """
    def __init__(self, p=0.25, sl=0.02, sh=0.2, r1=0.3, **kwargs):
        super().__init__(**kwargs)
        self.p  = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def call(self, inputs, training=None):
        if not training:
            return inputs

        # 1. Wrap your exact block into a single-image function
        def _apply_single(img):
            # Change 'inputs' to 'img' for your shape extractions
            h = tf.shape(img)[0]  
            w = tf.shape(img)[1]  
            c = tf.shape(img)[2]  
            img_dtype = img.dtype

            def apply_erasing():
                img_area    = tf.cast(h * w, tf.float32)
                target_area = tf.random.uniform([], self.sl, self.sh) * img_area
                aspect      = tf.random.uniform([], self.r1, 1.0 / self.r1)

                cut_h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect)),     tf.int32)
                cut_w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect)),     tf.int32)
                cut_h = tf.maximum(tf.minimum(cut_h, h - 1), 2)
                cut_w = tf.maximum(tf.minimum(cut_w, w - 1), 2)

                h1 = tf.cast(tf.random.uniform([], 0, tf.cast(h - cut_h, tf.float32)), tf.int32)
                w1 = tf.cast(tf.random.uniform([], 0, tf.cast(w - cut_w, tf.float32)), tf.int32)

                noise = tf.random.uniform(tf.stack([cut_h, cut_w, c]), 0.0, 1.0, dtype=img_dtype)

                pad = [[h1, h - h1 - cut_h], [w1, w - w1 - cut_w], [0, 0]]
                mask  = tf.pad(tf.zeros([cut_h, cut_w, c], dtype=img_dtype), pad,
                               constant_values=tf.cast(1.0, img_dtype))
                patch = tf.pad(noise, pad, constant_values=tf.cast(0.0, img_dtype))
                
                # Change 'inputs' to 'img' here as well
                return img * mask + patch * (tf.cast(1.0, img_dtype) - mask)

            return tf.cond(
                tf.random.uniform([]) > self.p,
                lambda: img, # Change 'inputs' to 'img'
                apply_erasing
            )

        # 2. Add this conditional block at the bottom of call() to handle 3D or 4D
        if inputs.shape.ndims == 4:
            return tf.map_fn(_apply_single, inputs, fn_output_signature=inputs.dtype)
        
        return _apply_single(inputs)
    

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"p": self.p, "sl": self.sl, "sh": self.sh, "r1": self.r1})
        return cfg

# ── Offline alignment ──────────────────────────────────────────────────────

def align_dataset_offline(raw_dir: str, out_dir: str):
    """
    Offline face detection and alignment using FaceDetector.
    Uses detect_largest for stability.
    Falls back to center crop if no face is found.
    """

    import os
    import glob
    import cv2
    from detector import get_detector

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    detector = get_detector()
    os.makedirs(out_dir, exist_ok=True)

    identities = [
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ]

    total_processed = 0
    skipped_corrupted = 0

    print(f"[dataset] Offline alignment: {raw_dir} → {out_dir}")

    for identity in identities:
        src_id = os.path.join(raw_dir, identity)
        dst_id = os.path.join(out_dir, identity)
        os.makedirs(dst_id, exist_ok=True)

        for img_path in glob.glob(os.path.join(src_id, "*.*")):
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                skipped_corrupted += 1
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                skipped_corrupted += 1
                continue

            total_processed += 1

            # ── FACE DETECTION ─────────────────────────────
            best = detector.detect_largest(img_bgr)

            if best is None:
                # fallback: center crop
                h, w = img_bgr.shape[:2]
                sz = min(h, w)
                x1, y1 = (w - sz) // 2, (h - sz) // 2
                crop = img_bgr[y1:y1 + sz, x1:x1 + sz]

                if crop.size == 0:
                    skipped_corrupted += 1
                    continue

                aligned = cv2.resize(
                    crop,
                    config.IMAGE_SIZE,
                    interpolation=cv2.INTER_CUBIC
                )
            else:
                # IMPORTANT: detect_largest usually returns face crop OR bbox+crop
                if isinstance(best, dict) and "face_crop" in best:
                    aligned = best["face_crop"]
                else:
                    # fallback: assume it's already an image crop
                    aligned = best

            cv2.imwrite(
                os.path.join(dst_id, os.path.basename(img_path)),
                aligned
            )

    print(f"[dataset] Done. Aligned: {total_processed}  Skipped/Corrupted: {skipped_corrupted}")

def prepare_aligned_dataset(
    raw_dir: str = None,
    out_dir: str = None,
) -> str:
    """
    Convenience wrapper called from training_demo.ipynb.
    Aligns the raw dataset and returns the output directory path.
    """
    raw = raw_dir or config.DATA_DIR
    out = out_dir or (config.DATA_DIR.rstrip("/\\") + "_aligned")
    align_dataset_offline(raw, out)
    return out


# ── Class catalogue helpers ────────────────────────────────────────────────

def build_class_catalogue(data_dir: str = None):
    """
    Scan `data_dir` and return (class_names, class_to_idx) where
    class_names is a sorted list of identity sub-folder names and
    class_to_idx maps each name to a consecutive integer index.

    Called by training_demo.ipynb cell 6 to inspect the dataset.
    """
    data_dir = data_dir or config.DATA_DIR
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return class_names, class_to_idx

# ── Augmentation pipeline ──────────────────────────────────────────────────

def get_augmentation_pipeline():
    """On-the-fly augmentation built from config parameters."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal") if config.AUGMENT_FLIP else layers.Layer(),
        layers.RandomRotation(factor=config.AUGMENT_ROTATION / 360.0, fill_mode="constant"),
        layers.RandomZoom(height_factor=config.AUGMENT_ZOOM, width_factor=config.AUGMENT_ZOOM,
                          fill_mode="constant"),
        layers.RandomBrightness(factor=config.AUGMENT_BRIGHTNESS),
        layers.RandomContrast(factor=config.AUGMENT_CONTRAST),
        RandomOcclusionErasing(p=0.25),
    ], name="data_augmentation")


# ── Image parse function ───────────────────────────────────────────────────

def _parse_function(filename, label):
    """Read → decode JPEG → resize → normalise to [0, 1]."""
    raw   = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(raw, channels=config.NUM_CHANNELS)
    image = tf.image.resize(image, config.IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
# ── Dataset builder ────────────────────────────────────────────────────────

def build_datasets(data_dir: str = config.DATA_DIR):
    """
    Scan data directory, filter sparse classes, split identities cleanly
    (80 / 10 / 10), and return optimised tf.data streams.

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
        Batched and prefetched pipelines.
    train_identities : list[str]
        Names of all training identities — acts as the class catalogue
        (position 4 so train.py can unpack as `class_names`).
    val_identities : list[str]
        Names of validation identities (for VerificationCallback).
    test_identities : list[str]
        Names of test identities (for final evaluation).
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found at: {data_dir}")

    all_identities = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    valid_identities   = []
    identity_to_images = {}

    print(f"[dataset] Scanning {data_dir}…")
    for identity in all_identities:
        idir = os.path.join(data_dir, identity)
        imgs = (glob.glob(os.path.join(idir, "*.[jJ][pP][gG]")) +
                glob.glob(os.path.join(idir, "*.[jJ][pP][eE][gG]")) +
                glob.glob(os.path.join(idir, "*.[pP][nN][gG]")))
        valid_imgs = [f for f in imgs
                      if os.path.exists(f) and os.path.getsize(f) > 0]
        if len(valid_imgs) >= config.MIN_IMAGES_PER_CLASS:
            valid_identities.append(identity)
            identity_to_images[identity] = sorted(valid_imgs)

    print(f"[dataset] Total: {len(all_identities)} | "
          f"Kept (≥{config.MIN_IMAGES_PER_CLASS} imgs): {len(valid_identities)}")

    if not valid_identities:
        raise ValueError(
            f"No identities passed the minimum of {config.MIN_IMAGES_PER_CLASS} images."
        )

    # ── [FIX-1] Identity-level split — no leakage ─────────────────────────
    rng = np.random.default_rng(config.RANDOM_SEED)
    shuffled = list(valid_identities)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * 0.80)
    n_val   = int(n_total * 0.10)

    train_identities = shuffled[:n_train]
    val_identities   = shuffled[n_train:n_train + n_val]
    test_identities  = shuffled[n_train + n_val:]

    print(f"[dataset] Splits: {len(train_identities)} train | "
          f"{len(val_identities)} val | {len(test_identities)} test identities")

    # Only training identities get classification labels
    # Only training identities get classification labels
    train_id_to_label = {ident: idx for idx, ident in enumerate(train_identities)}

    def _gather(identity_list, is_train=False):
        paths, labels = [], []
        for ident in identity_list:
            for img_path in identity_to_images[ident]:
                paths.append(img_path)
                labels.append(train_id_to_label[ident] if is_train else -1)
        return paths, labels

    train_paths, train_labels = _gather(train_identities, is_train=True)
    val_paths,   val_labels   = _gather(val_identities)
    test_paths,  test_labels  = _gather(test_identities)

# ── tf.data pipelines ─────────────────────────────────────────────────
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds   = tf.data.Dataset.from_tensor_slices((val_paths,   val_labels))
    test_ds  = tf.data.Dataset.from_tensor_slices((test_paths,  test_labels))

    train_ds = (
        train_ds
        .shuffle(buffer_size=len(train_paths), seed=config.RANDOM_SEED)
        .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds  = (val_ds
               .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(config.BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))
    test_ds = (test_ds
               .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(config.BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))

    # [FIX-5] Correct return order: train_identities in position 4 so
    # train.py can unpack it as `class_names` and compute num_classes correctly.
    return train_ds, val_ds, test_ds, train_identities, val_identities, test_identities

# ── Verification pair builder ──────────────────────────────────────────────

def build_verification_pairs(
    data_dir:   str,
    identities: list,
    num_pairs:  int = 2000,
):
    """
    Generate balanced verification pairs (50 % genuine, 50 % impostor)
    drawn ONLY from the provided identity list. [FIX-3]

    Args:
        data_dir:   Root directory of the face dataset.
        identities: Explicit list of identity names to draw from.
        num_pairs:  Total number of pairs (split equally between classes).

    Returns:
        paths1, paths2 : list[str]  — file paths for each pair.
        pair_labels    : list[int]  — 1 (genuine) or 0 (impostor).
    """
    rng = np.random.default_rng(config.RANDOM_SEED)

    class_images = {}
    for ident in identities:
        idir = os.path.join(data_dir, ident)
        imgs = (glob.glob(os.path.join(idir, "*.[jJ][pP][gG]")) +
                glob.glob(os.path.join(idir, "*.[jJ][pP][eE][gG]")) +
                glob.glob(os.path.join(idir, "*.[pP][nN][gG]")))
        class_images[ident] = sorted(
            [f for f in imgs if os.path.exists(f) and os.path.getsize(f) > 0]
        )

    paths1, paths2, pair_labels = [], [], []
    n_genuine_target = num_pairs // 2

    # ── Genuine pairs ─────────────────────────────────────────────────────
    per_identity = max(1, n_genuine_target // max(1, len(identities)))
    for ident in identities:
        imgs = class_images[ident]
        if len(imgs) < 2:
            continue
        max_possible = len(imgs) * (len(imgs) - 1) // 2
        target       = min(per_identity, max_possible)
        chosen: set  = set()
        attempts     = 0
        while len(chosen) < target and attempts < target * 10:
            i, j = rng.choice(len(imgs), size=2, replace=False)
            chosen.add((min(i, j), max(i, j)))
            attempts += 1
        for i, j in chosen:
            paths1.append(imgs[i])
            paths2.append(imgs[j])
            pair_labels.append(1)

    # ── Impostor pairs (balanced) ─────────────────────────────────────────
    n_genuine    = len(pair_labels)
    impostor_set: set = set()
    attempts     = 0
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

    return paths1, paths2, pair_labels