"""
BraTS2020 Preprocessing Script — EEE-515 Medical Imaging
=========================================================
Dataset: BraTS2020 brain MRI (pre-sliced .h5 format)
  - 369 volumes × 155 slices = 57,195 total .h5 files
  - image: (240, 240, 4) float64  — 4 modalities: T1, T1ce, T2, FLAIR
  - mask:  (240, 240, 3) uint8    — one-hot: WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)

Pipeline:
  1. Filter empty slices (all-zero image/mask)
  2. Clip & re-normalize each modality to [0, 1]
  3. Volume-level train/val/test split (70/15/15) — prevents data leakage
  4. Save split index CSVs to processed/brats/
"""

import os
import re
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\sbhat\Downloads\archive (3)\BraTS2020_training_data\content\data"
META_CSV   = os.path.join(DATA_DIR, "meta_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed", "brats")

# ─── Config ───────────────────────────────────────────────────────────────────
MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]
MASK_CHANNELS  = ["WT", "TC", "ET"]   # Whole Tumor, Tumor Core, Enhancing Tumor
IMG_SIZE       = (240, 240)           # spatial size (already correct in dataset)
CLIP_PERCENTILE = 99.5                # upper clip to remove bright outliers
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15
RANDOM_SEED    = 42


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_h5_slice(h5_path: str):
    """Load image and mask arrays from a single .h5 file.

    Returns:
        image: (240, 240, 4) float64 — 4 MRI modalities
        mask:  (240, 240, 3) uint8   — one-hot segmentation mask
    """
    with h5py.File(h5_path, "r") as f:
        image = f["image"][:]   # (H, W, 4)
        mask  = f["mask"][:]    # (H, W, 3)
    return image, mask


def is_empty_slice(image: np.ndarray, mask: np.ndarray,
                   min_nonzero_pixels: int = 100) -> bool:
    """Return True if the slice is essentially background (no brain tissue).

    A slice is considered empty when fewer than `min_nonzero_pixels` pixels
    are non-zero across all modalities.
    """
    return int((image != 0).any(axis=-1).sum()) < min_nonzero_pixels


def normalize_image(image: np.ndarray, clip_percentile: float = CLIP_PERCENTILE) -> np.ndarray:
    """Per-modality normalization to [0, 1].

    Steps:
      1. Clip upper outliers at `clip_percentile` (per modality, non-zero voxels only)
      2. Min-max scale to [0, 1] using the non-zero region statistics
      3. Clamp final values to [0, 1]

    Args:
        image: (H, W, 4) float array (may already be z-scored)
    Returns:
        normalized image: (H, W, 4) float32 in [0, 1]
    """
    image = image.astype(np.float32)
    normalized = np.zeros_like(image)

    for c in range(image.shape[-1]):
        channel = image[:, :, c]
        nonzero_vals = channel[channel != 0]

        if nonzero_vals.size == 0:
            # Blank modality channel — leave as zero
            continue

        # Clip upper outliers using non-zero region
        upper = np.percentile(nonzero_vals, clip_percentile)
        channel = np.clip(channel, 0, upper)

        # Min-max scale using non-zero region
        vmin = nonzero_vals.min()
        vmax = upper
        if vmax > vmin:
            channel = (channel - vmin) / (vmax - vmin)
        else:
            channel = np.zeros_like(channel)

        normalized[:, :, c] = np.clip(channel, 0.0, 1.0)

    return normalized


def get_volume_id(filename: str) -> str:
    """Extract volume ID from filename, e.g. 'volume_41_slice_0.h5' → '41'."""
    match = re.search(r"volume_(\d+)_slice", filename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot parse volume ID from: {filename}")


# ─── Main Preprocessing ───────────────────────────────────────────────────────

def build_split_index():
    """
    Create train/val/test split CSVs at the volume level.
    Each CSV row: slice_filename, volume_id, has_tumor (0/1).
    Saves to OUTPUT_DIR/train.csv, val.csv, test.csv.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading meta_data.csv …")
    meta = pd.read_csv(META_CSV)
    # meta columns: slice_path, target, volume, slice
    # slice_path is like '/content/data/volume_41_slice_0.h5'
    meta["filename"] = meta["slice_path"].apply(lambda p: os.path.basename(p))
    meta["volume_id"] = meta["volume"].astype(str)

    # ── Volume-level split (prevents train/val/test data leakage) ──
    unique_volumes = meta["volume_id"].unique().tolist()
    print(f"Total volumes: {len(unique_volumes)}")

    # Limit to first 10 volumes for testing
    unique_volumes = unique_volumes[:10]
    meta = meta[meta["volume_id"].isin(unique_volumes)].copy()
    print(f"Using first 10 volumes: {unique_volumes}")

    # First split off test set, then split remainder into train/val
    train_vols, temp_vols = train_test_split(
        unique_volumes, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
    )
    val_vols, test_vols = train_test_split(
        temp_vols, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
    )

    vol_to_split = (
        {v: "train" for v in train_vols} |
        {v: "val"   for v in val_vols}   |
        {v: "test"  for v in test_vols}
    )
    meta["split"] = meta["volume_id"].map(vol_to_split)

    for split_name in ["train", "val", "test"]:
        split_df = meta[meta["split"] == split_name].copy()
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
        split_df[["filename", "volume_id", "target"]].to_csv(out_path, index=False)
        print(f"  {split_name:5s}: {len(split_df):6d} slices  ({split_df['volume_id'].nunique()} volumes)  -> {out_path}")

    return meta


def filter_empty_slices(meta: pd.DataFrame,
                        min_nonzero_pixels: int = 100) -> pd.DataFrame:
    """
    Scan all .h5 files and flag which slices are non-empty (have brain tissue).
    Adds a boolean column 'has_brain' to meta.
    Saves updated CSVs (only non-empty rows) to OUTPUT_DIR/*_filtered.csv.

    Note: This scans 57k files — will take a few minutes.
    """
    print(f"\nFiltering empty slices (min_nonzero_pixels={min_nonzero_pixels}) …")
    has_brain = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Scanning"):
        h5_path = os.path.join(DATA_DIR, row["filename"])
        try:
            image, mask = load_h5_slice(h5_path)
            has_brain.append(not is_empty_slice(image, mask, min_nonzero_pixels))
        except Exception:
            has_brain.append(False)

    meta = meta.copy()
    meta["has_brain"] = has_brain

    total = len(meta)
    kept  = meta["has_brain"].sum()
    print(f"  Kept {kept}/{total} slices ({100*kept/total:.1f}% non-empty)")

    for split_name in ["train", "val", "test"]:
        split_df = meta[(meta["split"] == split_name) & meta["has_brain"]]
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}_filtered.csv")
        split_df[["filename", "volume_id", "target"]].to_csv(out_path, index=False)
        print(f"  {split_name:5s} filtered: {len(split_df):6d} slices")

    return meta


def verify_preprocessing(num_samples: int = 5):
    """
    Spot-check preprocessing on a few slices: load, normalize, and print stats.
    Verifies that normalize_image() produces values in [0, 1].
    """
    print(f"\nVerifying preprocessing on {num_samples} random non-empty slices …")

    # Use train_filtered.csv if it exists, else train.csv
    csv_path = os.path.join(OUTPUT_DIR, "train_filtered.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(OUTPUT_DIR, "train.csv")

    df = pd.read_csv(csv_path)
    sample = df.sample(n=min(num_samples, len(df)), random_state=RANDOM_SEED)

    for _, row in sample.iterrows():
        h5_path = os.path.join(DATA_DIR, row["filename"])
        image, mask = load_h5_slice(h5_path)
        image_norm = normalize_image(image)

        print(f"  {row['filename']}")
        print(f"    raw   image: [{image.min():7.3f}, {image.max():7.3f}]  mean={image.mean():.3f}")
        print(f"    norm  image: [{image_norm.min():7.3f}, {image_norm.max():7.3f}]  mean={image_norm.mean():.3f}")
        print(f"    mask  unique: {np.unique(mask).tolist()}  "
              f"ch_sums={[int(mask[:,:,c].sum()) for c in range(3)]}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BraTS2020 Preprocessing")
    print("=" * 60)

    # Step 1: Build volume-level train/val/test split index
    meta = build_split_index()

    # Step 2: Filter empty (all-zero) slices
    # Set run_filter=False to skip the full scan if CSVs already exist
    run_filter = not os.path.exists(os.path.join(OUTPUT_DIR, "train_filtered.csv"))
    if run_filter:
        meta = filter_empty_slices(meta, min_nonzero_pixels=100)
    else:
        print("\nFiltered CSVs already exist — skipping scan. Delete them to re-run.")

    # Step 3: Spot-check normalization
    verify_preprocessing(num_samples=5)

    print("\nDone. Output files written to:", OUTPUT_DIR)
    print("  train.csv / val.csv / test.csv          — all slices, volume-level split")
    print("  train_filtered.csv / ...                — empty slices removed")
