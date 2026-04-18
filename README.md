# EEE515 Medical Imaging — 3D Brain Tumor Visualization

![BraTS2020 rotating 3D tumor](demo.gif)

Interactive 3D segmentation visualization of glioma sub-regions from the
**BraTS2020** dataset using ground-truth masks, Marching Cubes surface
reconstruction, and PyVista rendering.

---

## Tumor regions

| Color | Region | BraTS Label |
|-------|--------|-------------|
| Red | Enhancing tumor | 4 |
| Yellow | Necrotic core | 1 |
| Green | Peritumoral edema | 2 |

---

## Interactive viewer

```bash
python visualize_brats3d.py
```

| Key | Action |
|-----|--------|
| Left-click drag | Rotate |
| Scroll | Zoom |
| N / Right arrow | Next patient |
| P / Left arrow | Previous patient |
| E | Toggle edema |
| C | Toggle necrotic core |
| T | Toggle enhancing tumor |
| R | Reset camera |
| Q | Quit |

---

## Generate demo GIF

```bash
python visualize_brats3d.py --gif
```

---

## Requirements

```
pip install pyvista scikit-image h5py nibabel imageio numpy matplotlib
```

---

## Data

BraTS2020 training data (Kaggle):
`archive (3).zip` — 369 patient volumes, 155 axial slices each,
stored as `.h5` files with 4 MRI modalities + 3-channel binary segmentation mask.

---

## Pipeline

1. Stacks 155 axial `.h5` slices per patient into a `(240, 240, 155)` label volume
2. Runs **Marching Cubes** (`skimage`) on each tumor sub-region independently
3. Laplacian-smooths the surface meshes (15 iterations)
4. Renders with **PyVista** / VTK with per-region color and opacity

---

*EEE-515 Medical Imaging — Arizona State University*
