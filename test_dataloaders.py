"""
Smoke test for both DataLoaders -- EEE-515 Medical Imaging
Verifies tensor shapes, dtypes, and value ranges for one batch each.
"""

import torch
from datasets import get_brats_loaders, get_monuseg_loaders


def check_batch(name, images, masks):
    print(f"  image  : {tuple(images.shape)}  dtype={images.dtype}"
          f"  range=[{images.min():.3f}, {images.max():.3f}]")
    print(f"  mask   : {tuple(masks.shape)}   dtype={masks.dtype}"
          f"  unique={torch.unique(masks).tolist()[:6]}")
    assert images.dtype == torch.float32, "image must be float32"
    assert masks.dtype  == torch.float32, "mask must be float32"
    assert not torch.isnan(images).any(), "NaN in images!"
    assert not torch.isnan(masks).any(),  "NaN in masks!"
    print("  [OK] no NaN, correct dtype")


print("=" * 55)
print("BraTS2020 DataLoaders")
print("=" * 55)
brats_train, brats_val, brats_test = get_brats_loaders(batch_size=8)

print("\nTrain batch:")
imgs, masks, meta = next(iter(brats_train))
check_batch("BraTS train", imgs, masks)
print(f"  meta   : volume_ids={meta['volume_id'][:3]}, has_tumor={meta['has_tumor'][:3]}")

print("\nVal batch:")
imgs, masks, _ = next(iter(brats_val))
check_batch("BraTS val", imgs, masks)

print("\nTest batch:")
imgs, masks, _ = next(iter(brats_test))
check_batch("BraTS test", imgs, masks)


print()
print("=" * 55)
print("MoNuSeg DataLoaders")
print("=" * 55)
mono_train, mono_val, mono_test = get_monuseg_loaders(batch_size=4)

print("\nTrain batch:")
imgs, masks, meta = next(iter(mono_train))
check_batch("MoNuSeg train", imgs, masks)
print(f"  meta   : {meta['img_filename']}")

print("\nVal batch:")
imgs, masks, _ = next(iter(mono_val))
check_batch("MoNuSeg val", imgs, masks)

print("\nTest batch:")
imgs, masks, _ = next(iter(mono_test))
check_batch("MoNuSeg test", imgs, masks)

print()
print("All checks passed.")
