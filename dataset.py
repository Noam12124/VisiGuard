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
    so train.py can pass them to the verification matching callback.
"""

import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import config


# ── Custom Augmentation Layer for Occlusion ────────────────────────────────

class RandomOcclusionErasing(layers.Layer):
    """
    Simulates real-world occlusions (masks, sunglasses, hats) by randomly 
    erasing a rectangular patch from the face image to improve robustness.
    """
    def __init__(self, p=0.25, sl=0.02, sh=0.2, r1=0.3, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        c = tf.shape(inputs)[3]
        
        def erase_single_img(img):
            if tf.random.uniform([]) > self.p:
                return img
            
            img_area = tf.cast(h * w, tf.float32)
            target_area = tf.random.uniform([], self.sl, self.sh) * img_area
            aspect_ratio = tf.random.uniform([], self.r1, 1 / self.r1)
            
            cut_h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
            cut_w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)
            
            cut_h = tf.minimum(cut_h, h - 1)
            cut_w = tf.minimum(cut_w, w - 1)
            
            cut_h = tf.maximum(cut_h, 2)
            cut_w = tf.maximum(cut_w, 2)
            
            h1 = tf.cast(tf.random.uniform([], 0, tf.cast(h - cut_h, tf.float32)), tf.int32)
            w1 = tf.cast(tf.random.uniform([], 0, tf.cast(w - cut_w, tf.float32)), tf.int32)
            
            noise = tf.random.uniform(tf.stack([cut_h, cut_w, c]), 0.0, 1.0)
            
            padding_top = h1
            padding_bottom = h - h1 - cut_h
            padding_left = w1
            padding_right = w - w1 - cut_w
            
            patch_mask = tf.pad(
                tf.zeros([cut_h, cut_w, c]),
                [[padding_top, padding_bottom], [padding_left, padding_right], [0, 0]],
                constant_values=1.0
            )
            patch_noise = tf.pad(
                noise,
                [[padding_top, padding_bottom], [padding_left, padding_right], [0, 0]],
                constant_values=0.0
            )
            
            return img * patch_mask + patch_noise * (1.0 - patch_mask)

        return tf.map_fn(erase_single_img, inputs, fn_output_signature=tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "p": self.p,
            "sl": self.sl,
            "sh": self.sh,
            "r1": self.r1
        })
        return cfg


# ── Data Augmentation Pipeline ─────────────────────────────────────────────

def get_augmentation_pipeline():
    """Builds an on-the-fly image augmentation sequential block using configuration parameters."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal") if config.AUGMENT_FLIP else layers.Layer(),
        layers.RandomRotation(factor=config.AUGMENT_ROTATION / 360.0, fill_mode="constant"),
        layers.RandomZoom(height_factor=config.AUGMENT_ZOOM, width_factor=config.AUGMENT_ZOOM, fill_mode="constant"),
        layers.RandomBrightness(factor=config.AUGMENT_BRIGHTNESS),
        layers.RandomContrast(factor=config.AUGMENT_CONTRAST),
        RandomOcclusionErasing(p=0.25)
    ], name="data_augmentation")


# ── Parse and Load Image Function for tf.data ─────────────────────────────

def _parse_function(filename, label):
    """Reads image file, decodes, resizes, and normalizes safely."""
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=config.NUM_CHANNELS)
    image = tf.image.resize(image, config.IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# ── Offline Dataset Alignment Hardening ────────────────────────────────────

def align_dataset_offline(raw_dir: str, out_dir: str):
    """
    Performs offline face detection and alignment using YOLOv8-face.
    [FIX-4] Hardened to catch race conditions and corrupted/empty images.
    """
    from detector import get_detector
    
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        
    detector = get_detector()
    os.makedirs(out_dir, exist_ok=True)
    
    identities = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    skipped_corrupted = 0
    total_processed = 0
    
    print(f"[dataset] Starting offline alignment from {raw_dir} to {out_dir}...")
    
    for identity in identities:
        src_id_dir = os.path.join(raw_dir, identity)
        dst_id_dir = os.path.join(out_dir, identity)
        os.makedirs(dst_id_dir, exist_ok=True)
        
        imgs = glob.glob(os.path.join(src_id_dir, "*.*"))
        for img_path in imgs:
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            # [FIX-4] Guard against empty file allocations in cloud drives
            if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                skipped_corrupted += 1
                continue
                
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                skipped_corrupted += 1
                continue
                
            total_processed += 1
            detections = detector.detect_faces(img_bgr, conf_threshold=config.FACE_CONF_THRESHOLD)
            
            if len(detections) == 0:
                h, w = img_bgr.shape[:2]
                sz = min(h, w)
                x1, y1 = (w - sz) // 2, (h - sz) // 2
                crop = img_bgr[y1:y1+sz, x1:x1+sz]
                if crop.size > 0:
                    aligned = cv2.resize(crop, config.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
                else:
                    skipped_corrupted += 1
                    continue
            else:
                best_det = max(detections, key=lambda x: x["confidence"])
                aligned = best_det["face_crop"]
                
            out_path = os.path.join(dst_id_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, aligned)
            
    print(f"[dataset] Alignment complete. Aligned: {total_processed} images. Skipped/Corrupted: {skipped_corrupted}")


# ── Identity Disjoint Splitting and Dataset Creation ───────────────────────

def build_datasets(data_dir: str = config.DATA_DIR):
    """
    Scans data directory, filters sparse classes, splits identities cleanly (80/10/10),
    and sets up optimized training, validation, and testing tf.data streams.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found at: {data_dir}")

    all_identities = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    valid_identities = []
    identity_to_images = {}

    print(f"[dataset] Scanning identities in {data_dir}...")
    
    # [FIX-2] Filter identities by MIN_IMAGES_PER_CLASS
    for identity in all_identities:
        idir = os.path.join(data_dir, identity)
        imgs = glob.glob(os.path.join(idir, "*.[jJ][pP][gG]")) + \
               glob.glob(os.path.join(idir, "*.[jJ][pP][eE][gG]")) + \
               glob.glob(os.path.join(idir, "*.[sS][vV][gG]")) + \
               glob.glob(os.path.join(idir, "*.[pP][nN][gG]"))
        
        valid_imgs = [f for f in imgs if os.path.exists(f) and os.path.getsize(f) > 0]
        if len(valid_imgs) >= config.MIN_IMAGES_PER_CLASS:
            valid_identities.append(identity)
            identity_to_images[identity] = sorted(valid_imgs)

    print(f"[dataset] Total identities: {len(all_identities)} | Kept (≥{config.MIN_IMAGES_PER_CLASS} images): {len(valid_identities)}")

    if not valid_identities:
        raise ValueError(f"Zero identities passed the minimum requirement of {config.MIN_IMAGES_PER_CLASS} images.")

    # [FIX-1] Split on identity level first to guarantee zero data leakage
    rng = np.random.default_rng(config.RANDOM_SEED)
    shuffled_identities = list(valid_identities)
    rng.shuffle(shuffled_identities)

    n_total = len(shuffled_identities)
    n_train = int(n_total * 0.80)
    n_val   = int(n_total * 0.10)

    train_identities = shuffled_identities[:n_train]
    val_identities   = shuffled_identities[n_train:n_train + n_val]
    test_identities  = shuffled_identities[n_train + n_val:]

    print(f"[dataset] Identity Splits: {len(train_identities)} train | {len(val_identities)} val | {len(test_identities)} test")

    # Map only training identities to continuous categorical classes for classification loss
    train_id_to_label = {identity: idx for idx, identity in enumerate(train_identities)}

    def gather_split_paths_and_labels(identity_list, is_train=False):
        paths, labels = [], []
        for identity in identity_list:
            for img_path in identity_to_images[identity]:
                paths.append(img_path)
                labels.append(train_id_to_label[identity] if is_train else -1)
        return paths, labels

    train_paths, train_labels = gather_split_paths_and_labels(train_identities, is_train=True)
    val_paths, val_labels     = gather_split_paths_and_labels(val_identities, is_train=False)
    test_paths, test_labels   = gather_split_paths_and_labels(test_identities, is_train=False)

    # Building tf.data pipelines
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds   = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    test_ds  = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

    train_ds = train_ds.shuffle(buffer_size=len(train_paths), seed=config.RANDOM_SEED)
    train_ds = train_ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    # On-the-fly augmentation execution
    aug_pipeline = get_augmentation_pipeline()
    train_ds = train_ds.batch(config.BATCH_SIZE)
    train_ds = train_ds.map(lambda x, y: (aug_pipeline(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = val_ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # [FIX-5] Returns val and test identity sets for proper callback validation tracking
    return train_ds, val_ds, test_ds, val_identities, test_identities, len(train_identities)


# ── Verification Pairs Construction ────────────────────────────────────────

def build_verification_pairs(data_dir: str, identities: list, num_pairs: int = 2000):
    """
    Generates balanced verification image pairs (50% genuine, 50% impostor)
    drawn EXCLUSIVELY from the provided identity list [FIX-3].
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    
    class_images = {}
    for identity in identities:
        idir = os.path.join(data_dir, identity)
        imgs = glob.glob(os.path.join(idir, "*.[jJ][pP][gG]")) + \
               glob.glob(os.path.join(idir, "*.[jJ][pP][eE][gG]")) + \
               glob.glob(os.path.join(idir, "*.[pP][nN][gG]"))
        class_images[identity] = sorted([f for f in imgs if os.path.exists(f) and os.path.getsize(f) > 0])

    paths1, paths2, pair_labels = [], [], []
    n_genuine_target = num_pairs // 2

    # ── Genuine pairs (Same person) ───────────────────────────────────────
    per_identity = max(1, n_genuine_target // len(identities))
    
    for identity in identities:
        imgs = class_images[identity]
        if len(imgs) < 2:
            continue
        
        chosen = set()
        attempts = 0
        max_possible = len(imgs) * (len(imgs) - 1) // 2
        target = min(per_identity, max_possible)
        
        while len(chosen) < target and attempts < target * 10:
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

    return paths1, paths2, pair_labels