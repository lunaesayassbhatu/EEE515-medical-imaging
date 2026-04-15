"""
MoNuSeg (kmms) Preprocessing Script -- EEE-515 Medical Imaging
===============================================================
Dataset: Multi-organ Nucleus Segmentation (TCGA H&E microscopy images)
  Training: 24 images (1000x1000 RGB .tif) + 24 masks (256x256 instance .png)
  Test:     58 images (256x256 RGBA .png or 1000x1000 RGB) + 58 masks (256x256)

Key observations:
  - Masks are INSTANCE labels (each nucleus = unique integer, 0 = background)
  - Training images are 1000x1000 but masks are 256x256 -> resize images to match
  - Test images may have 4 channels (RGBA) -> drop alpha
  - Mask filenames have trailing spaces (e.g. 'TCGA-... .png')
  - For segmentation: convert instance masks to BINARY (0=bg, 1=nucleus)

Pipeline:
  1. Load image (RGB) + instance mask
  2. Resize image to 256x256 to match mask spatial size
  3. Convert RGBA -> RGB if needed
  4. Convert instance mask to binary mask (0/1)
  5. Normalize image to [0, 1] with ImageNet mean/std (standard for H&E)
  6. Volume-level train/val split (80/20) from 24 training images
  7. Save split index CSVs to processed/monuseg/
"""

import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Paths -------------------------------------------------------------------
TRAIN_IMG_DIR  = r"C:\Users\sbhat\Downloads\archive (4)\kmms_training\kmms_training\images"
TRAIN_MASK_DIR = r"C:\Users\sbhat\Downloads\archive (4)\kmms_training\kmms_training\masks"
TEST_IMG_DIR   = r"C:\Users\sbhat\Downloads\archive (4)\kmms_test\kmms_test\images"
TEST_MASK_DIR  = r"C:\Users\sbhat\Downloads\archive (4)\kmms_test\kmms_test\masks"
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "processed", "monuseg")

# --- Config ------------------------------------------------------------------
TARGET_SIZE  = (256, 256)      # resize all images to this (matches mask size)
TRAIN_RATIO  = 0.80            # 80% train, 20% val from the 24 training images
RANDOM_SEED  = 42

# ImageNet mean/std for H&E RGB normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --- Helpers -----------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load image as (H, W, 3) uint8 RGB, handling RGBA and grayscale."""
    img = Image.open(path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def load_mask(path: str) -> np.ndarray:
    """Load instance segmentation mask as (H, W) uint8."""
    msk = Image.open(path.strip())   # strip() handles trailing spaces in filenames
    return np.array(msk)


def instance_to_binary(mask: np.ndarray) -> np.ndarray:
    """Convert instance label mask to binary: 0=background, 1=nucleus."""
    return (mask > 0).astype(np.uint8)


def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """Resize (H, W, C) image to size=(H', W') using high-quality Lanczos."""
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize((size[1], size[0]), Image.LANCZOS)
    return np.array(pil_img)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize (H, W, 3) uint8 image to float32 using ImageNet mean/std.

    Returns float32 array in approximately [-2.1, 2.6] range (standard for
    pretrained CNN backbones on H&E histology).
    """
    img = image.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def match_mask_filename(img_name: str, mask_dir: str) -> str:
    """Find the corresponding mask file for a given image name.

    Handles trailing spaces in MoNuSeg mask filenames (e.g. 'TCGA-... .png').
    """
    stem = os.path.splitext(img_name)[0]
    # Try exact match first, then with trailing space
    for candidate in [f"{stem}.png", f"{stem} .png", f"{stem}  .png"]:
        if os.path.exists(os.path.join(mask_dir, candidate)):
            return candidate
    # Fallback: search directory
    for f in os.listdir(mask_dir):
        if f.strip() == f"{stem}.png":
            return f
    raise FileNotFoundError(f"No mask found for image: {img_name}")


# --- Build split index -------------------------------------------------------

def build_split_index():
    """Create train/val split CSVs from the 24 training images.

    Each CSV row: img_filename, mask_filename, split
    Test set is kept as a separate fixed CSV.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_imgs = sorted(os.listdir(TRAIN_IMG_DIR))
    print(f"Training images found: {len(train_imgs)}")

    # Pair each image with its mask
    pairs = []
    for img_name in train_imgs:
        mask_name = match_mask_filename(img_name, TRAIN_MASK_DIR)
        pairs.append((img_name, mask_name))

    # Train/val split (image-level, since each image is independent)
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED
    )

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["img_filename", "mask_filename"])
            writer.writerows(split_pairs)
        print(f"  {split_name:5s}: {len(split_pairs):3d} images -> {out_path}")

    # Test set index
    test_imgs  = sorted(os.listdir(TEST_IMG_DIR))
    test_masks = sorted(os.listdir(TEST_MASK_DIR))
    out_path = os.path.join(OUTPUT_DIR, "test.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_filename", "mask_filename"])
        for img_f, msk_f in zip(test_imgs, test_masks):
            writer.writerow([img_f, msk_f])
    print(f"  {'test':5s}: {len(test_imgs):3d} images -> {out_path}")


# --- Verify preprocessing ----------------------------------------------------

def verify_preprocessing(num_samples: int = 4):
    """Spot-check the full preprocessing pipeline on a few training images."""
    print(f"\nVerifying preprocessing on {num_samples} training samples ...")

    train_imgs = sorted(os.listdir(TRAIN_IMG_DIR))[:num_samples]

    for img_name in train_imgs:
        img_path  = os.path.join(TRAIN_IMG_DIR, img_name)
        mask_name = match_mask_filename(img_name, TRAIN_MASK_DIR)
        mask_path = os.path.join(TRAIN_MASK_DIR, mask_name)

        # Load
        image = load_image(img_path)
        mask  = load_mask(mask_path)

        # Process
        image_resized = resize_image(image, TARGET_SIZE)
        binary_mask   = instance_to_binary(mask)
        image_norm    = normalize_image(image_resized)

        nucleus_px  = int(binary_mask.sum())
        total_px    = binary_mask.size
        nucleus_pct = 100 * nucleus_px / total_px

        print(f"\n  {img_name}")
        print(f"    raw image:    {image.shape}  range [{image.min()}, {image.max()}]")
        print(f"    resized:      {image_resized.shape}")
        print(f"    normalized:   [{image_norm.min():.3f}, {image_norm.max():.3f}]"
              f"  mean={image_norm.mean():.3f}")
        print(f"    instance mask unique labels: {np.unique(mask).shape[0]}  "
              f"(max={mask.max()})")
        print(f"    binary mask:  nucleus coverage = {nucleus_pct:.1f}%  "
              f"({nucleus_px}/{total_px} px)")


def verify_test_preprocessing(num_samples: int = 3):
    """Spot-check preprocessing on a few test images."""
    print(f"\nVerifying test set preprocessing on {num_samples} samples ...")

    test_imgs  = sorted(os.listdir(TEST_IMG_DIR))[:num_samples]
    test_masks = sorted(os.listdir(TEST_MASK_DIR))[:num_samples]

    for img_name, mask_name in zip(test_imgs, test_masks):
        img_path  = os.path.join(TEST_IMG_DIR, img_name)
        mask_path = os.path.join(TEST_MASK_DIR, mask_name)

        raw = Image.open(img_path)
        image = load_image(img_path)
        mask  = load_mask(mask_path)

        image_resized = resize_image(image, TARGET_SIZE) if image.shape[:2] != TARGET_SIZE else image
        binary_mask   = instance_to_binary(mask)
        image_norm    = normalize_image(image_resized)

        print(f"\n  {img_name.strip()} (original mode: {raw.mode})")
        print(f"    raw image:   {image.shape}  range [{image.min()}, {image.max()}]")
        print(f"    normalized:  [{image_norm.min():.3f}, {image_norm.max():.3f}]")
        print(f"    binary mask: {binary_mask.shape}  nucleus coverage = "
              f"{100*binary_mask.mean():.1f}%")


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MoNuSeg (kmms) Preprocessing")
    print("=" * 60)
    print(f"Target image size : {TARGET_SIZE}")
    print(f"Mask type         : instance -> binary (nucleus vs background)")
    print(f"Normalization     : ImageNet mean/std")
    print()

    # Step 1: Build split index CSVs
    build_split_index()

    # Step 2: Spot-check training preprocessing
    verify_preprocessing(num_samples=4)

    # Step 3: Spot-check test preprocessing
    verify_test_preprocessing(num_samples=3)

    print("\nDone. Output files written to:", OUTPUT_DIR)
    print("  train.csv  -- 80% of 24 training images")
    print("  val.csv    -- 20% of 24 training images")
    print("  test.csv   -- all 58 test images (fixed split)")
    print()
    print("Preprocessing applied at load time (no files saved to disk):")
    print("  resize_image()        -- 1000x1000 -> 256x256 (Lanczos)")
    print("  instance_to_binary()  -- instance labels -> 0/1 binary mask")
    print("  normalize_image()     -- [0,255] -> ImageNet normalized float32")
