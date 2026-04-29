import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 20)
ax.axis('off')
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')

# ── colour palette ──────────────────────────────────────────────────────────
C_HEADER   = '#161B22'
C_BOX_EDGE = {'data': '#1F6FEB', 'proc': '#238636', 'out': '#A371F7', 'view': '#F78166'}
C_BOX_FACE = {'data': '#0D2045', 'proc': '#0D2618', 'out': '#1E0A40', 'view': '#2D0A07'}
C_ARROW    = '#8B949E'
C_TEXT     = '#E6EDF3'
C_SUB      = '#8B949E'
C_BADGE    = '#30363D'
C_TITLE    = '#58A6FF'

stages = [
    {
        'label': '1   INPUT DATA',
        'title': 'BraTS 2020 Dataset',
        'subtitle': 'Multi-modal Brain Tumor MRI',
        'details': ['• 4 MRI modalities: T1, T1ce, T2, FLAIR',
                    '• 369 training patients',
                    '• NIfTI format (.nii.gz)',
                    '• Voxel resolution: 1 mm³ isotropic'],
        'badge': 'NIfTI  ·  4-channel  ·  240×240×155',
        'kind': 'data',
        'y': 18.1,
    },
    {
        'label': '2   PREPROCESSING',
        'title': 'Z-Score Normalization',
        'subtitle': 'Per-channel intensity standardisation',
        'details': ['• μ = 0, σ = 1 per modality per patient',
                    '• Brain-mask applied (non-zero voxels)',
                    '• Cropped to 128×128×128 patches',
                    '• PyTorch tensor conversion'],
        'badge': 'MONAI  ·  torchio  ·  NumPy',
        'kind': 'proc',
        'y': 13.9,
    },
    {
        'label': '3   SEGMENTATION',
        'title': 'SegResNet',
        'subtitle': 'Residual encoder–decoder architecture',
        'details': ['• MONAI SegResNet (depth 6)',
                    '• Input: 4-channel 128³ patch',
                    '• Output: 3-channel probability map',
                    '• Trained with Dice + CE loss'],
        'badge': 'MONAI  ·  PyTorch  ·  CUDA',
        'kind': 'proc',
        'y': 9.7,
    },
    {
        'label': '4   SEGMENTATION LABELS',
        'title': 'Tumour Sub-region Masks',
        'subtitle': 'Three hierarchical tumour labels',
        'details': ['• ET — Enhancing Tumour (class 3)',
                    '• TC — Tumour Core  (ET + necrosis)',
                    '• WT — Whole Tumour (TC + oedema)',
                    '• Binary voxel masks per sub-region'],
        'badge': 'ET  ·  TC  ·  WT',
        'kind': 'out',
        'y': 5.5,
    },
    {
        'label': '5   3D RECONSTRUCTION  &  VISUALISATION',
        'title': 'Marching Cubes  →  PyVista Viewer',
        'subtitle': 'Surface mesh extraction & interactive rendering',
        'details': ['• scikit-image marching_cubes (level=0.5)',
                    '• Smooth surface meshes per sub-region',
                    '• PyVista plotter — rotation, slicing, opacity',
                    '• Exported as .png / .vtk / .glb'],
        'badge': 'scikit-image  ·  PyVista  ·  VTK',
        'kind': 'view',
        'y': 1.3,
    },
]

BOX_W   = 11.0
BOX_H   = 3.5
BOX_X   = 1.5
ARROW_H = 0.55

def draw_stage(ax, s):
    kind = s['kind']
    y    = s['y']
    ec   = C_BOX_EDGE[kind]
    fc   = C_BOX_FACE[kind]

    # shadow
    shadow = FancyBboxPatch(
        (BOX_X + 0.07, y - BOX_H - 0.07), BOX_W, BOX_H,
        boxstyle='round,pad=0.15', linewidth=0,
        facecolor='#000000', alpha=0.55, zorder=1)
    ax.add_patch(shadow)

    # main box
    box = FancyBboxPatch(
        (BOX_X, y - BOX_H), BOX_W, BOX_H,
        boxstyle='round,pad=0.15', linewidth=2,
        edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)

    # left accent bar
    bar = FancyBboxPatch(
        (BOX_X, y - BOX_H), 0.22, BOX_H,
        boxstyle='round,pad=0.0', linewidth=0,
        facecolor=ec, alpha=0.9, zorder=3)
    ax.add_patch(bar)

    # stage label (top-left)
    ax.text(BOX_X + 0.40, y - 0.30, s['label'],
            color=ec, fontsize=8, fontweight='bold',
            va='top', ha='left', zorder=4,
            fontfamily='monospace')

    # title
    ax.text(BOX_X + 0.40, y - 0.70, s['title'],
            color=C_TEXT, fontsize=13, fontweight='bold',
            va='top', ha='left', zorder=4)

    # subtitle
    ax.text(BOX_X + 0.40, y - 1.10, s['subtitle'],
            color=C_SUB, fontsize=9, style='italic',
            va='top', ha='left', zorder=4)

    # separator line
    ax.plot([BOX_X + 0.35, BOX_X + BOX_W - 0.20], [y - 1.35, y - 1.35],
            color=ec, alpha=0.35, linewidth=0.8, zorder=4)

    # detail bullets
    for i, det in enumerate(s['details']):
        ax.text(BOX_X + 0.45, y - 1.55 - i * 0.44, det,
                color=C_TEXT, fontsize=8.5, va='top', ha='left', zorder=4,
                fontfamily='monospace')

    # badge pill (bottom-right)
    bx = BOX_X + BOX_W - 0.20
    by = y - BOX_H + 0.28
    badge = FancyBboxPatch(
        (bx - len(s['badge']) * 0.068 - 0.18, by - 0.17),
        len(s['badge']) * 0.068 + 0.36, 0.36,
        boxstyle='round,pad=0.05', linewidth=1,
        edgecolor=ec, facecolor=C_BADGE, alpha=0.85, zorder=4)
    ax.add_patch(badge)
    ax.text(bx - len(s['badge']) * 0.034, by,
            s['badge'], color=ec, fontsize=7.5, fontweight='bold',
            va='center', ha='center', zorder=5, fontfamily='monospace')


def draw_arrow(ax, y_top, y_bot, label=''):
    mid_x = BOX_X + BOX_W / 2
    y0 = y_top - ARROW_H * 0.2
    y1 = y_bot + ARROW_H * 0.2
    ax.annotate('', xy=(mid_x, y1), xytext=(mid_x, y0),
                arrowprops=dict(
                    arrowstyle='->', color=C_ARROW,
                    lw=2.0,
                    connectionstyle='arc3,rad=0'))
    if label:
        ax.text(mid_x + 0.25, (y0 + y1) / 2, label,
                color=C_SUB, fontsize=8, va='center', ha='left',
                style='italic')


# ── title block ─────────────────────────────────────────────────────────────
ax.text(7, 19.65, 'EEE 515 — Medical Imaging',
        color=C_TITLE, fontsize=11, fontweight='bold',
        va='top', ha='center', fontfamily='monospace')
ax.text(7, 19.30, 'Brain Tumour Segmentation & 3-D Reconstruction Pipeline',
        color=C_TEXT, fontsize=15, fontweight='bold',
        va='top', ha='center')
ax.plot([1.0, 13.0], [19.05, 19.05], color=C_TITLE, alpha=0.40, linewidth=1)

# ── draw stages ─────────────────────────────────────────────────────────────
for s in stages:
    draw_stage(ax, s)

# ── arrows between stages ───────────────────────────────────────────────────
conn_labels = ['tensor / batch', 'probability maps', 'binary masks', 'surface meshes']
for i in range(len(stages) - 1):
    y_top_box = stages[i]['y'] - BOX_H       # bottom edge of box i
    y_bot_box = stages[i + 1]['y']           # top edge of box i+1
    mid_x = BOX_X + BOX_W / 2
    label_y = (y_top_box + y_bot_box) / 2

    # single arrow
    ax.annotate('', xy=(mid_x, y_bot_box + 0.06),
                xytext=(mid_x, y_top_box - 0.06),
                arrowprops=dict(arrowstyle='->', color='#FFFFFF', lw=2.5),
                zorder=6)

    # dark pill behind label for contrast
    pad_x, pad_y = 0.28, 0.18
    txt_w = len(conn_labels[i]) * 0.092
    pill = FancyBboxPatch(
        (mid_x - txt_w / 2 - pad_x, label_y - pad_y),
        txt_w + pad_x * 2, pad_y * 2,
        boxstyle='round,pad=0.05', linewidth=1.2,
        edgecolor='#FFFFFF', facecolor='#161B22', alpha=0.92, zorder=7)
    ax.add_patch(pill)

    # bold white label centred on the arrow
    ax.text(mid_x, label_y, conn_labels[i],
            color='#FFFFFF', fontsize=11, fontweight='bold',
            va='center', ha='center', zorder=8)

# ── footer ───────────────────────────────────────────────────────────────────
ax.plot([1.0, 13.0], [0.50, 0.50], color=C_TITLE, alpha=0.25, linewidth=0.8)
ax.text(7, 0.38, 'BraTS 2020  |  SegResNet  |  MONAI  |  PyVista  |  scikit-image  |  PyTorch',
        color=C_SUB, fontsize=7.5, va='top', ha='center', fontfamily='monospace')

plt.tight_layout(pad=0.3)
plt.savefig(
    r'C:\Users\sbhat\PycharmProjects\EEE515_medical_imaging\pipeline_diagram.png',
    dpi=180, bbox_inches='tight',
    facecolor=fig.get_facecolor())
print("Saved pipeline_diagram.png")
plt.close()
