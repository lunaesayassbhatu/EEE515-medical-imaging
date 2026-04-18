"""
segment_brats.py
================
Run the pretrained MONAI SegResNet on BraTS2020 h5 data.

Pipeline
--------
1. Reconstruct 4-channel 3D MRI volumes from 2D h5 slices
2. Z-score normalize each MRI channel on non-zero voxels (matches training)
3. Run SlidingWindowInferer (ROI 240x240x160, overlap 0.5)
4. Post-process: sigmoid -> threshold(0.5) -> BraTS label map
5. Compare predictions to ground-truth masks
6. Report ET / TC / WT Dice scores
7. Save predicted label maps (.npy) for 3D visualization

BraTS regions
-------------
  ET  Enhancing Tumor   label 4  (red)
  TC  Tumor Core        labels 1+4
  WT  Whole Tumor       labels 1+2+4

Model output channels
---------------------
  ch 0 -> Necrotic Core   (label 1)
  ch 1 -> Edema           (label 2)
  ch 2 -> Enhancing Tumor (label 4)

Usage
-----
  python segment_brats.py            # all 10 patients
  python segment_brats.py --n 1      # 1 patient for a quick smoke-test (~3 min CPU)
  python segment_brats.py --n 3      # 3 patients
"""

import argparse
import io
import json
import os
import re
import time
import zipfile

import h5py
import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import SegResNet
from monai.transforms import NormalizeIntensity

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
ZIP_PATH    = r"C:\Users\sbhat\Downloads\archive (3).zip"
BUNDLE_DIR  = r"C:\Users\sbhat\PycharmProjects\EEE515_medical_imaging\pretrained\brats_mri_segmentation"
RESULTS_DIR = r"C:\Users\sbhat\PycharmProjects\EEE515_medical_imaging\seg_results"
N_PATIENTS  = 10

# Matches pretrained bundle's inference.json exactly
ROI_SIZE   = (240, 240, 160)
SW_OVERLAP = 0.5
THRESHOLD  = 0.5

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
def load_model(device):
    """Instantiate SegResNet and load pretrained weights."""
    model = SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)

    weights_path = os.path.join(BUNDLE_DIR, "models", "model.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found at {weights_path}.\n"
            "Run: python -c \"from monai.bundle import download; "
            "download(name='brats_mri_segmentation', bundle_dir='./pretrained')\""
        )

    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    # Bundle saves as {"model": state_dict} or just state_dict
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded weights: {weights_path}")
    return model


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def get_volume_ids(zf, n=10):
    seen = {}
    for name in zf.namelist():
        m = re.search(r"volume_(\d+)_slice_", name)
        if m:
            seen[int(m.group(1))] = True
        if len(seen) >= n:
            break
    return sorted(seen.keys())[:n]


def load_patient_3d(zf, vol_id):
    """
    Stack h5 slices into:
      image_4d : (4, 240, 240, 155)  float32   -- 4 MRI modalities
      gt_labels: (240, 240, 155)     uint8     -- BraTS label map (0/1/2/4)
    """
    pattern = re.compile(rf"volume_{vol_id}_slice_(\d+)\.h5$")
    slice_map = {}
    for name in zf.namelist():
        m = pattern.search(name)
        if m:
            slice_map[int(m.group(1))] = name

    if not slice_map:
        return None, None

    n_slices = max(slice_map.keys()) + 1
    image_buf = np.zeros((240, 240, n_slices, 4), dtype=np.float32)
    mask_buf  = np.zeros((240, 240, n_slices),    dtype=np.uint8)

    for s_idx in sorted(slice_map.keys()):
        with zf.open(slice_map[s_idx]) as raw:
            buf = io.BytesIO(raw.read())
        with h5py.File(buf, "r") as hf:
            image_buf[:, :, s_idx, :] = hf["image"][:].astype(np.float32)
            mask                       = hf["mask"][:]   # (240,240,3) binary

        # Convert 3-channel binary mask to BraTS label convention
        lbl = np.zeros((240, 240), dtype=np.uint8)
        lbl[mask[:, :, 1] == 1] = 2   # edema
        lbl[mask[:, :, 0] == 1] = 1   # necrotic (overwrites edema)
        lbl[mask[:, :, 2] == 1] = 4   # enhancing (highest priority)
        mask_buf[:, :, s_idx] = lbl

    # Rearrange to (4, H, W, D)
    image_4d = image_buf.transpose(3, 0, 1, 2)   # (4, 240, 240, 155)
    return image_4d, mask_buf


def normalize_image(image_4d):
    """
    Per-channel z-score normalization on non-zero voxels only.
    Matches the MONAI bundle's NormalizeIntensityd(nonzero=True, channel_wise=True).
    """
    norm = NormalizeIntensity(nonzero=True, channel_wise=True)
    tensor = torch.from_numpy(image_4d)   # (4, H, W, D)
    return norm(tensor).numpy()


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------
def run_inference(model, image_4d, device):
    """
    image_4d : (4, H, W, D)  normalized float32
    Returns  : (3, H, W, D)  sigmoid probability maps
    """
    inp = torch.from_numpy(image_4d).unsqueeze(0).to(device)  # (1,4,H,W,D)

    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        sw_batch_size=1,
        overlap=SW_OVERLAP,
        mode="gaussian",
        sigma_scale=0.125,
    )

    with torch.no_grad():
        logits = inferer(inp, model)   # (1, 3, H, W, D)

    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (3, H, W, D)
    return probs


def probs_to_labels(probs):
    """
    Convert 3-channel sigmoid probabilities to BraTS label map.
    ch0 = Necrotic Core  -> label 1
    ch1 = Edema          -> label 2
    ch2 = Enhancing      -> label 4
    Priority: enhancing > necrotic > edema (matching bundle's Lambda)
    """
    pred = np.zeros(probs.shape[1:], dtype=np.uint8)
    pred[probs[1] > THRESHOLD] = 2   # edema
    pred[probs[0] > THRESHOLD] = 1   # necrotic (overwrites edema)
    pred[probs[2] > THRESHOLD] = 4   # enhancing (highest priority)
    return pred


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def dice(pred_mask, gt_mask):
    inter = int((pred_mask & gt_mask).sum())
    denom = int(pred_mask.sum()) + int(gt_mask.sum())
    return (2.0 * inter / denom) if denom > 0 else 1.0   # 1.0 if both empty


def compute_region_dice(pred, gt):
    """
    Returns dict with ET, TC, WT Dice scores.
    pred, gt: (H, W, D) uint8 label maps
    """
    return {
        "ET": dice(pred == 4,             gt == 4),
        "TC": dice((pred == 1) | (pred == 4),  (gt == 1) | (gt == 4)),
        "WT": dice(pred > 0,              gt > 0),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BraTS segmentation with MONAI SegResNet")
    parser.add_argument("--n", type=int, default=N_PATIENTS,
                        help="Number of patients to process (default: 10)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 62)
    print("BraTS2020 -- AI Segmentation with MONAI SegResNet")
    print("=" * 62)
    print(f"  Device  : {device}")
    print(f"  Patients: {args.n}")
    print(f"  ROI     : {ROI_SIZE}  overlap={SW_OVERLAP}")
    if device.type == "cpu":
        print("  [NOTE] CPU mode: ~2-5 min per patient. Use GPU for 10x speed.")
    print()

    # Load model
    print("Loading pretrained SegResNet ...")
    model = load_model(device)
    print()

    # Gather volume IDs
    with zipfile.ZipFile(ZIP_PATH) as zf:
        vol_ids = get_volume_ids(zf, n=args.n)
        print(f"Processing volumes: {vol_ids}\n")

        all_results = []
        t_total = time.time()

        for i, vid in enumerate(vol_ids, 1):
            print(f"[{i}/{len(vol_ids)}] volume_{vid}")
            t0 = time.time()

            # 1) Load 3D volume from h5 slices
            print(f"  Loading {vid} slices from zip ...")
            image_4d, gt_labels = load_patient_3d(zf, vid)
            if image_4d is None:
                print("  [!] No slices found -- skipping")
                continue

            gt_vox = {lv: int((gt_labels == lv).sum()) for lv in [1, 2, 4]}
            print(f"  GT voxels -- NC:{gt_vox[1]:,}  edema:{gt_vox[2]:,}  ET:{gt_vox[4]:,}")

            # 2) Normalize
            print("  Normalizing ...")
            image_norm = normalize_image(image_4d)

            # 3) Inference
            print(f"  Running SegResNet inference (ROI {ROI_SIZE}) ...")
            probs = run_inference(model, image_norm, device)

            # 4) Post-process to label map
            pred_labels = probs_to_labels(probs)
            pred_vox = {lv: int((pred_labels == lv).sum()) for lv in [1, 2, 4]}
            print(f"  Pred voxels -- NC:{pred_vox[1]:,}  edema:{pred_vox[2]:,}  ET:{pred_vox[4]:,}")

            # 5) Dice scores
            scores = compute_region_dice(pred_labels, gt_labels)
            elapsed = time.time() - t0
            print(f"  Dice -- ET:{scores['ET']:.3f}  TC:{scores['TC']:.3f}  "
                  f"WT:{scores['WT']:.3f}  ({elapsed:.0f}s)")

            # 6) Save predicted label map
            out_path = os.path.join(RESULTS_DIR, f"pred_volume_{vid}.npy")
            np.save(out_path, pred_labels)

            # Also save confidence maps for inspection
            conf_path = os.path.join(RESULTS_DIR, f"probs_volume_{vid}.npy")
            np.save(conf_path, probs.astype(np.float16))  # half-precision to save space

            all_results.append({
                "vol_id": vid,
                "dice_ET": round(scores["ET"], 4),
                "dice_TC": round(scores["TC"], 4),
                "dice_WT": round(scores["WT"], 4),
                "gt_voxels":   gt_vox,
                "pred_voxels": pred_vox,
                "elapsed_sec": round(elapsed, 1),
            })
            print()

    # Summary
    total_time = time.time() - t_total
    print("=" * 62)
    print(f"Results for {len(all_results)} patients  ({total_time/60:.1f} min total)")
    print("=" * 62)
    print(f"{'Vol':>7}  {'ET':>6}  {'TC':>6}  {'WT':>6}")
    print("-" * 35)
    for r in all_results:
        print(f"{r['vol_id']:>7}  {r['dice_ET']:>6.3f}  {r['dice_TC']:>6.3f}  {r['dice_WT']:>6.3f}")

    if all_results:
        mean_et = np.mean([r["dice_ET"] for r in all_results])
        mean_tc = np.mean([r["dice_TC"] for r in all_results])
        mean_wt = np.mean([r["dice_WT"] for r in all_results])
        print("-" * 35)
        print(f"  Mean  {mean_et:>6.3f}  {mean_tc:>6.3f}  {mean_wt:>6.3f}")

    # Save results JSON
    summary_path = os.path.join(RESULTS_DIR, "dice_scores.json")
    with open(summary_path, "w") as f:
        json.dump({"patients": all_results,
                   "mean": {"ET": round(float(mean_et), 4),
                             "TC": round(float(mean_tc), 4),
                             "WT": round(float(mean_wt), 4)} if all_results else {}},
                  f, indent=2)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  Label maps  : pred_volume_*.npy")
    print(f"  Confidence  : probs_volume_*.npy")
    print(f"  Dice scores : dice_scores.json")
    print()
    print("To view AI predictions in 3D:")
    print("  python visualize_brats3d.py --predicted")


if __name__ == "__main__":
    main()
