"""
download_lfw.py — LFW dataset loader using TensorFlow Datasets (stable, no broken URLs)

Keeps SAME pipeline architecture (extract_and_organise unchanged).
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

import config
import tensorflow_datasets as tfds


LFW_EXTRACT_DIR  = "/tmp/lfw_raw_tfds_fake"


def download_with_tfds():
    """
    Download LFW safely using TensorFlow Datasets.
    Returns a folder structured like:
        /tmp/lfw_raw_tfds_fake/
            person_id/
                img.jpg
    """
    print("Downloading LFW via TensorFlow Datasets...")

    ds = tfds.load("lfw", split="train", as_supervised=False)

    os.makedirs(LFW_EXTRACT_DIR, exist_ok=True)

    counts = {}

    for sample in tfds.as_numpy(ds):
        image = sample["image"]
        label = sample["label"]

        # ✅ FIX: label is bytes like b'George_W_Bush'
        label = label.decode("utf-8")

        person_dir = os.path.join(LFW_EXTRACT_DIR, label)
        os.makedirs(person_dir, exist_ok=True)

        counts[label] = counts.get(label, 0) + 1
        img_path = os.path.join(person_dir, f"{counts[label]}.jpg")

        with open(img_path, "wb") as f:
            f.write(image)

    print("TFDS download + conversion done.")

def extract_and_organise(
    archive_path: str,
    extract_dir: str,
    out_dir: str,
    min_images: int,
):
    """
    SAME FUNCTION AS YOUR ORIGINAL (UNCHANGED LOGIC)
    Works directly on TFDS-generated folder.
    """

    print(f"Using dataset at: {extract_dir}")

    lfw_root = extract_dir

    identities = sorted(os.listdir(lfw_root))

    os.makedirs(out_dir, exist_ok=True)
    kept = 0
    skipped = 0

    for identity in identities:
        src_folder = os.path.join(lfw_root, identity)
        if not os.path.isdir(src_folder):
            continue

        images = [
            f for f in os.listdir(src_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) < min_images:
            skipped += 1
            continue

        dst_folder = os.path.join(out_dir, identity)
        os.makedirs(dst_folder, exist_ok=True)

        for img in images:
            src = os.path.join(src_folder, img)
            dst = os.path.join(dst_folder, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        kept += 1

    print(f"\nDone!  {kept} identities kept  (≥{min_images} images each)")
    print(f"       {skipped} identities skipped")
    print(f"\nDataset ready at: {out_dir}")

    return kept


def parse_args():
    p = argparse.ArgumentParser(description="Download LFW using TFDS (stable version)")
    p.add_argument("--min-images", type=int, default=config.MIN_IMAGES_PER_CLASS)
    p.add_argument("--out", default=config.DATA_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    # STEP 1: download via TFDS (NO URLS, NO CRASHES)
    download_with_tfds()

    # STEP 2: use SAME pipeline function you already had
    extract_and_organise(
        archive_path=None,
        extract_dir=LFW_EXTRACT_DIR,
        out_dir=args.out,
        min_images=args.min_images,
    )

    print("\nNext step: python train.py")


if __name__ == "__main__":
    main()