import zipfile, h5py, io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

ZIP   = r'C:\Users\sbhat\Downloads\archive (3).zip'
SLICE = 'BraTS2020_training_data/content/data/volume_100_slice_77.h5'
OUT1  = r'C:\Users\sbhat\PycharmProjects\EEE515_medical_imaging\sample_data_figure.png'
OUT2  = r'C:\Users\sbhat\PycharmProjects\EEE515_medical_imaging\segmentation_masks.png'

# ── load data ────────────────────────────────────────────────────────────────
with zipfile.ZipFile(ZIP) as z:
    with z.open(SLICE) as f:
        raw = f.read()

with h5py.File(io.BytesIO(raw), 'r') as h:
    img  = h['image'][:]   # (240, 240, 4) – T1, T1ce, T2, FLAIR  (z-scored)
    mask = h['mask'][:]    # (240, 240, 3) – ch0=WT, ch1=TC, ch2=ET (binary)

# ── reconstruct original BraTS labels from sub-region masks ──────────────────
# ET  = ch2  (label 4 – enhancing tumour)
# TC  = ch1  (label 1+4 – tumour core)
# WT  = ch0  (label 1+2+4 – whole tumour)
# Necrotic (label 1) = TC  AND NOT ET
# Edema    (label 2) = WT  AND NOT TC
et        = mask[:, :, 2].astype(bool)           # label 4
necrotic  = mask[:, :, 1].astype(bool) & ~et     # label 1
edema     = mask[:, :, 0].astype(bool) & ~mask[:, :, 1].astype(bool)  # label 2

# composite label image: 0=bg, 1=necrotic(yellow), 2=edema(green), 4=ET(red)
label_img = np.zeros((240, 240), dtype=np.uint8)
label_img[edema]    = 2
label_img[necrotic] = 1
label_img[et]       = 4

def norm(arr):
    """clip top 1% then min-max to [0,1]"""
    p99 = np.percentile(arr[arr > 0], 99) if arr.max() > 0 else 1.0
    out = np.clip(arr, arr.min(), p99)
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo + 1e-8)

T1    = norm(img[:, :, 0])
T1ce  = norm(img[:, :, 1])
T2    = norm(img[:, :, 2])
FLAIR = norm(img[:, :, 3])

# overlay: colour each label on top of greyscale T1ce
def make_overlay(base, lbl):
    rgb = np.stack([base, base, base], axis=-1)
    rgb[lbl == 1] = [1.0, 1.0, 0.0]   # yellow – necrotic
    rgb[lbl == 2] = [0.0, 0.8, 0.0]   # green  – edema
    rgb[lbl == 4] = [1.0, 0.15, 0.15] # red    – ET
    return rgb

overlay = make_overlay(T1ce, label_img)

# ── shared style ──────────────────────────────────────────────────────────────
BG   = '#0D1117'
TITLE_C = '#58A6FF'
LABEL_C = '#E6EDF3'

# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – sample_data_figure.png  (2 rows)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
fig.patch.set_facecolor(BG)
for ax in axes.flat:
    ax.set_facecolor(BG)
    ax.axis('off')

fig.suptitle(
    'BraTS 2020 — Patient 100, Axial Slice 77\n'
    'Multi-Modal MRI & Tumour Segmentation Overlay',
    color=TITLE_C, fontsize=14, fontweight='bold', y=0.98)

# ── ROW 1 : raw modalities ───────────────────────────────────────────────────
row1 = [
    (T1,    'T1'),
    (T1ce,  'T1ce\n(Contrast-Enhanced)'),
    (T2,    'T2'),
    (FLAIR, 'FLAIR'),
]
for col, (arr, title) in enumerate(row1):
    ax = axes[0, col]
    ax.imshow(arr, cmap='gray', origin='upper', vmin=0, vmax=1)
    ax.set_title(title, color=LABEL_C, fontsize=11, fontweight='bold', pad=6)

axes[0, 0].set_ylabel('Row 1: MRI Modalities',
                       color=TITLE_C, fontsize=10, fontweight='bold',
                       labelpad=8)

# ── ROW 2 : segmentation overlay  (same image 4 ×, progressive labels) ──────
panels = [
    (make_overlay(T1ce, (label_img == 4).astype(np.uint8) * 4),
     'Enhancing Tumor\n(Label 4 – Red)'),
    (make_overlay(T1ce, (label_img == 1).astype(np.uint8) * 1),
     'Necrotic Core\n(Label 1 – Yellow)'),
    (make_overlay(T1ce, (label_img == 2).astype(np.uint8) * 2),
     'Edema\n(Label 2 – Green)'),
    (overlay,
     'All Sub-regions\nOverlaid on T1ce'),
]
for col, (arr, title) in enumerate(panels):
    ax = axes[1, col]
    ax.imshow(arr, origin='upper')
    ax.set_title(title, color=LABEL_C, fontsize=10, fontweight='bold', pad=6)

axes[1, 0].set_ylabel('Row 2: Segmentation Overlay',
                       color=TITLE_C, fontsize=10, fontweight='bold',
                       labelpad=8)

# legend on final overlay panel
legend_patches = [
    mpatches.Patch(color='red',    label='ET – Enhancing Tumour (Label 4)'),
    mpatches.Patch(color='yellow', label='NCR – Necrotic Core (Label 1)'),
    mpatches.Patch(color='green',  label='ED – Peritumoral Edema (Label 2)'),
]
axes[1, 3].legend(
    handles=legend_patches, loc='lower left',
    fontsize=7.5, framealpha=0.75,
    facecolor='#161B22', edgecolor='#30363D',
    labelcolor=LABEL_C)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT1, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved {OUT1}')
plt.close()

# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 – segmentation_masks.png  (masks only, 3 + 1 panels)
# ════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4.5))
fig2.patch.set_facecolor(BG)
for ax in axes2:
    ax.set_facecolor(BG)
    ax.axis('off')

fig2.suptitle(
    'BraTS 2020 — Segmentation Masks  |  Patient 100, Slice 77',
    color=TITLE_C, fontsize=13, fontweight='bold', y=1.02)

# individual binary masks with colour maps
mask_panels = [
    (et.astype(np.float32),       'Enhancing Tumor (ET)\nLabel 4',      'Reds'),
    (necrotic.astype(np.float32), 'Necrotic Core (NCR)\nLabel 1',       'YlOrBr'),
    (edema.astype(np.float32),    'Peritumoral Edema (ED)\nLabel 2',     'Greens'),
    (None,                        'All Sub-regions\n(Colour-coded)',      None),
]

for col, (arr, title, cmap) in enumerate(mask_panels):
    ax = axes2[col]
    if arr is not None:
        ax.imshow(arr, cmap=cmap, origin='upper', vmin=0, vmax=1)
    else:
        # composite colour image
        composite = np.zeros((240, 240, 3), dtype=np.float32)
        composite[et]       = [1.0, 0.2, 0.2]
        composite[necrotic] = [1.0, 1.0, 0.0]
        composite[edema]    = [0.0, 0.8, 0.0]
        ax.imshow(composite, origin='upper')
    ax.set_title(title, color=LABEL_C, fontsize=10, fontweight='bold', pad=6)

    # pixel counts as subtitle
    if arr is not None:
        count = int(arr.sum())
    else:
        count = int(et.sum() + necrotic.sum() + edema.sum())
    ax.text(0.5, -0.04, f'{count} voxels', transform=ax.transAxes,
            color='#8B949E', fontsize=8, ha='center')

plt.tight_layout()
plt.savefig(OUT2, dpi=180, bbox_inches='tight', facecolor=fig2.get_facecolor())
print(f'Saved {OUT2}')
plt.close()
