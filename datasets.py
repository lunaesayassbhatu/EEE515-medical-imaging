"""
PyTorch DataLoaders -- EEE-515 Medical Imaging
===============================================
Provides Dataset classes and DataLoader factory functions for:
  - BraTSDataset   : brain MRI segmentation (4-channel .h5 slices)
  - MoNuSegDataset : nucleus segmentation (RGB H&E .tif images)

Usage:
    from datasets import get_brats_loaders, get_monuseg_loaders

    brats_train, brats_val, brats_test = get_brats_loaders(batch_size=16)
    mono_train,  mono_val,  mono_test  = get_monuseg_loaders(batch_size=8)
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Preprocessing functions (imported from preprocessing scripts) ─────────────
# BraTS
BRATS_DATA_DIR  = r"C:\Users\sbhat\Downloads\archive (3)\BraTS2020_training_data\content\data"
BRATS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "processed", "brats")

# MoNuSeg
MONO_TRAIN_IMG_DIR  = r"C:\Users\sbhat\Downloads\archive (4)\kmms_training\kmms_training\images"
MONO_TRAIN_MASK_DIR = r"C:\Users\sbhat\Downloads\archive (4)\kmms_training\kmms_training\masks"
MONO_TEST_IMG_DIR   = r"C:\Users\sbhat\Downloads\archive (4)\kmms_test\kmms_test\images"
MONO_TEST_MASK_DIR  = r"C:\Users\sbhat\Downloads\archive (4)\kmms_test\kmms_test\masks"
MONO_INDEX_DIR      = os.path.join(os.path.dirname(__file__), "processed", "monuseg")

MONO_TARGET_SIZE   = (256, 256)
MONO_IMAGENET_MEAN = (0.485, 0.456, 0.406)
MONO_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────────────────
# BraTS2020 Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _brats_normalize(image: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    """Per-modality clip + min-max normalize to [0, 1]. Returns float32."""
    image = image.astype(np.float32)
    out = np.zeros_like(image)
    for c in range(image.shape[-1]):
        ch = image[:, :, c]
        nz = ch[ch != 0]
        if nz.size == 0:
            continue
        upper = np.percentile(nz, clip_percentile)
        ch = np.clip(ch, 0.0, upper)
        vmin, vmax = nz.min(), upper
        if vmax > vmin:
            ch = (ch - vmin) / (vmax - vmin)
        out[:, :, c] = np.clip(ch, 0.0, 1.0)
    return out


def _brats_augmentations(split: str) -> A.Compose:
    """Albumentations pipeline for BraTS (multi-channel MRI).

    Train: random flips, 90-degree rotations, slight brightness/contrast shift.
    Val/Test: no augmentation.
    All: convert HWC numpy -> CHW tensor.
    """
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.1, p=0.3),
            ToTensorV2(),          # (H, W, C) -> (C, H, W), dtype preserved
        ])
    return A.Compose([ToTensorV2()])


class BraTSDataset(Dataset):
    """PyTorch Dataset for BraTS2020 pre-sliced .h5 data.

    Each item returns:
        image : FloatTensor (4, 240, 240) -- T1, T1ce, T2, FLAIR in [0, 1]
        mask  : FloatTensor (3, 240, 240) -- WT, TC, ET binary one-hot masks
        meta  : dict with filename, volume_id, has_tumor flag
    """

    def __init__(self, split: str = "train", augment: bool = True):
        """
        Args:
            split   : one of 'train', 'val', 'test'
            augment : if True, apply training augmentations (ignored for val/test)
        """
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        # Prefer filtered CSV (empty slices removed); fall back to full CSV
        filtered_csv = os.path.join(BRATS_INDEX_DIR, f"{split}_filtered.csv")
        full_csv     = os.path.join(BRATS_INDEX_DIR, f"{split}.csv")
        csv_path = filtered_csv if os.path.exists(filtered_csv) else full_csv

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Index CSV not found: {csv_path}\n"
                f"Run preprocess_brats.py first."
            )

        self.df       = pd.read_csv(csv_path)
        self.split    = split
        self.aug      = _brats_augmentations(split if augment else "val")
        self.data_dir = BRATS_DATA_DIR

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        h5_path = os.path.join(self.data_dir, row["filename"])

        with h5py.File(h5_path, "r") as f:
            image = f["image"][:]    # (240, 240, 4) float64
            mask  = f["mask"][:]     # (240, 240, 3) uint8

        # Normalize image modalities to [0, 1]
        image = _brats_normalize(image)           # (240, 240, 4) float32

        # Albumentations expects uint8 mask; pass as additional target
        # We keep mask as float32 after transform
        augmented = self.aug(image=image, mask=mask.astype(np.float32))
        image_t = augmented["image"].float()      # (4, 240, 240)
        mask_t  = augmented["mask"]               # (240, 240, 3) still HWC from albu

        # ToTensorV2 only converts 'image'; permute mask manually
        mask_t = mask_t.permute(2, 0, 1).float()  # (3, 240, 240)

        meta = {
            "filename":  row["filename"],
            "volume_id": str(row["volume_id"]),
            "has_tumor": int(row["target"]),
        }
        return image_t, mask_t, meta


# ─────────────────────────────────────────────────────────────────────────────
# MoNuSeg Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _mono_augmentations(split: str) -> A.Compose:
    """Albumentations pipeline for MoNuSeg RGB images.

    Images are pre-resized to 256x256 before this pipeline runs, so no
    Resize transform is needed here (avoids the shape-consistency check).

    Train: flips, rotations, color jitter (common for H&E stain variation),
           elastic deformation (mimics tissue deformation).
    Val/Test: only normalize + tensorize.
    """
    normalize = A.Normalize(mean=MONO_IMAGENET_MEAN, std=MONO_IMAGENET_STD)

    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1, p=0.4),
            A.ElasticTransform(alpha=30, sigma=5, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            normalize,
            ToTensorV2(),
        ])
    return A.Compose([normalize, ToTensorV2()])


class MoNuSegDataset(Dataset):
    """PyTorch Dataset for MoNuSeg (kmms) nucleus segmentation.

    Each item returns:
        image : FloatTensor (3, 256, 256) -- ImageNet-normalized RGB
        mask  : FloatTensor (1, 256, 256) -- binary nucleus mask (0/1)
        meta  : dict with img_filename
    """

    def __init__(self, split: str = "train", augment: bool = True):
        """
        Args:
            split   : one of 'train', 'val', 'test'
            augment : if True, apply training augmentations (ignored for val/test)
        """
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        csv_path = os.path.join(MONO_INDEX_DIR, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Index CSV not found: {csv_path}\n"
                f"Run preprocess_monuseg.py first."
            )

        self.df    = pd.read_csv(csv_path)
        self.split = split
        self.aug   = _mono_augmentations(split if augment else "val")

        # Point to the right image/mask directories
        if split in ("train", "val"):
            self.img_dir  = MONO_TRAIN_IMG_DIR
            self.mask_dir = MONO_TRAIN_MASK_DIR
        else:
            self.img_dir  = MONO_TEST_IMG_DIR
            self.mask_dir = MONO_TEST_MASK_DIR

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image as (H, W, 3) uint8 RGB, dropping alpha if present."""
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    def _load_mask(self, path: str) -> np.ndarray:
        """Load instance mask and convert to binary (H, W) uint8."""
        msk = np.array(Image.open(path.strip()))
        return (msk > 0).astype(np.uint8)   # instance -> binary

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path  = os.path.join(self.img_dir,  row["img_filename"])
        mask_path = os.path.join(self.mask_dir, row["mask_filename"])

        image = self._load_image(img_path)    # (H, W, 3) uint8
        mask  = self._load_mask(mask_path)    # (H, W)    uint8 binary

        # Resize image to TARGET_SIZE before albumentations to ensure
        # image and mask have matching spatial dimensions (mask is already 256x256)
        if image.shape[:2] != MONO_TARGET_SIZE:
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize(
                (MONO_TARGET_SIZE[1], MONO_TARGET_SIZE[0]), Image.LANCZOS
            )
            image = np.array(pil_img)

        augmented = self.aug(image=image, mask=mask)
        image_t = augmented["image"].float()           # (3, 256, 256)
        mask_t  = augmented["mask"].unsqueeze(0).float()  # (1, 256, 256)

        meta = {"img_filename": row["img_filename"]}
        return image_t, mask_t, meta


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────────────────────────────

def get_brats_loaders(
    batch_size: int = 16,
    num_workers: int = 0,
    augment_train: bool = True,
):
    """Return (train_loader, val_loader, test_loader) for BraTS2020.

    Args:
        batch_size    : samples per batch
        num_workers   : parallel data loading workers (0 = main process)
        augment_train : apply augmentations to training set

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = BraTSDataset(split="train", augment=augment_train)
    val_ds   = BraTSDataset(split="val",   augment=False)
    test_ds  = BraTSDataset(split="test",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"BraTS DataLoaders ready:")
    print(f"  train : {len(train_ds):5d} slices | {len(train_loader):4d} batches (bs={batch_size})")
    print(f"  val   : {len(val_ds):5d} slices | {len(val_loader):4d} batches")
    print(f"  test  : {len(test_ds):5d} slices | {len(test_loader):4d} batches")

    return train_loader, val_loader, test_loader


def get_monuseg_loaders(
    batch_size: int = 8,
    num_workers: int = 0,
    augment_train: bool = True,
):
    """Return (train_loader, val_loader, test_loader) for MoNuSeg.

    Args:
        batch_size    : samples per batch
        num_workers   : parallel data loading workers (0 = main process)
        augment_train : apply augmentations to training set

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = MoNuSegDataset(split="train", augment=augment_train)
    val_ds   = MoNuSegDataset(split="val",   augment=False)
    test_ds  = MoNuSegDataset(split="test",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"MoNuSeg DataLoaders ready:")
    print(f"  train : {len(train_ds):3d} images | {len(train_loader):3d} batches (bs={batch_size})")
    print(f"  val   : {len(val_ds):3d} images | {len(val_loader):3d} batches")
    print(f"  test  : {len(test_ds):3d} images | {len(test_loader):3d} batches")

    return train_loader, val_loader, test_loader
