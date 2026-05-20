"""
download_lfw.py — Download and organise the LFW dataset for training.

LFW (Labeled Faces in the Wild) gives ~1,680 identities with ≥5 images
each (~10,000 images total) and is freely downloadable — no access
request needed.

For production accuracy (93-97%) use VGGFace2:
    https://github.com/ox-vgg/vgg_face2

Usage:
    python download_lfw.py
    python download_lfw.py --min-images 10     # stricter filter
    python download_lfw.py --out data/my_lfw   # custom output dir
"""

import os
import sys
import argparse
import tarfile
import shutil
import urllib.request
from pathlib import Path

import config


LFW_URL = "https://github.com/robertwgh/LFW-dataset/releases/download/v1.0/lfw.tgz"
LFW_FUNNELED_URL = "https://github.com/robertwgh/LFW-dataset/releases/download/v1.0/lfw-funneled.tgz"
LFW_ARCHIVE      = "/tmp/lfw.tgz"
LFW_EXTRACT_DIR  = "/tmp/lfw_raw"


def download_with_progress(url: str, dest: str):
    """Download a file with a progress bar."""
    print(f"Downloading {url}…")
    downloaded = [0]
    total      = [0]

    def reporthook(count, block_size, total_size):
        downloaded[0] = count * block_size
        total[0]      = total_size
        if total_size > 0:
            pct = min(100, downloaded[0] * 100 // total_size)
            mb  = downloaded[0] / 1_048_576
            print(f"\r  {pct:3d}%  {mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def extract_and_organise(
    archive_path:  str,
    extract_dir:   str,
    out_dir:       str,
    min_images:    int,
):
    """
    Extract LFW archive and copy images into per-identity sub-folders,
    filtering identities with fewer than min_images images.
    """
    # Extract
    print(f"Extracting {archive_path}…")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    # LFW extracts to lfw/ or lfw_funneled/ inside extract_dir
    candidates = [
        os.path.join(extract_dir, "lfw"),
        os.path.join(extract_dir, "lfw_funneled"),
    ]
    lfw_root = next((c for c in candidates if os.path.isdir(c)), None)
    if lfw_root is None:
        # Try one level deeper
        for d in os.listdir(extract_dir):
            full = os.path.join(extract_dir, d)
            if os.path.isdir(full):
                lfw_root = full
                break

    if lfw_root is None:
        raise RuntimeError(f"Could not find LFW root inside {extract_dir}")

    print(f"LFW root: {lfw_root}")
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
    print(f"       {skipped} identities skipped (< {min_images} images)")
    print(f"\nDataset ready at: {out_dir}")
    return kept


def parse_args():
    p = argparse.ArgumentParser(description="Download and organise LFW dataset.")
    p.add_argument("--min-images", type=int, default=config.MIN_IMAGES_PER_CLASS)
    p.add_argument("--out",        default=config.DATA_DIR)
    p.add_argument("--funneled",   action="store_true",
                   help="Download the deep-funneled (pre-aligned) version.")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download if archive already exists.")
    return p.parse_args()


def main():
    args = parse_args()

    url = LFW_FUNNELED_URL if args.funneled else LFW_URL

    # Download
    if not args.skip_download or not os.path.exists(LFW_ARCHIVE):
        download_with_progress(url, LFW_ARCHIVE)
    else:
        print(f"Archive already at {LFW_ARCHIVE}; skipping download.")

    # Extract + organise
    n = extract_and_organise(
        archive_path = LFW_ARCHIVE,
        extract_dir  = LFW_EXTRACT_DIR,
        out_dir      = args.out,
        min_images   = args.min_images,
    )

    print(f"\nNext step:  python train.py")
    print(f"Or run the notebook:  notebooks/training_demo.ipynb")


if __name__ == "__main__":
    main()
