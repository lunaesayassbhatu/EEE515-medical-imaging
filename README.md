# EEE515 — 3D Brain Tumor Visualization

<p align="center">
  <img src="demo.gif" width="640" alt="Rotating 3D brain tumor visualization"/>
</p>

<p align="center">
  <b>Interactive 3D segmentation of glioma sub-regions from BraTS2020</b><br>
  Arizona State University &nbsp;|&nbsp; EEE-515 Medical Imaging &nbsp;|&nbsp; Spring 2025
</p>

---

## Project Description

This project reconstructs and visualizes the three glioma sub-regions defined
by the BraTS2020 challenge directly from ground-truth segmentation masks.
Each patient's 155 axial MRI slices are stacked into a 3D label volume;
Marching Cubes then extracts a surface mesh per region, which PyVista renders
interactively in a rotating 3D window.

The viewer loads all 10 patients at startup and lets you navigate between them
instantly with keyboard shortcuts, toggle individual tumor regions on/off, and
inspect per-region volume measurements in real time.

---

## Features

- **3D surface reconstruction** via Marching Cubes (`scikit-image`) on each
  tumor sub-region independently
- **Color-coded tumor anatomy**

  | Color | Region | BraTS Label |
  |-------|--------|-------------|
  | Red | Enhancing tumor | 4 |
  | Yellow | Necrotic / non-enhancing core | 1 |
  | Green | Peritumoral edema | 2 |

- **Interactive PyVista window** — rotate, zoom, and pan with the mouse
- **Multi-patient navigation** — browse all 10 patients without reloading
- **Per-region toggle** — show/hide each tumor zone with a key press or checkbox
- **Live volume readout** — mm³ and cm³ measurements updated per patient
- **Animated GIF export** — 72-frame 360-degree orbit rendered off-screen
- **Static PNG screenshots** — saved for all 10 patients plus a mosaic overview

---

## How to Run

### Requirements

```bash
pip install pyvista scikit-image h5py nibabel imageio numpy matplotlib
```

### Interactive 3D viewer

```bash
cd EEE515_medical_imaging
python visualize_brats3d.py
```

The script reads the BraTS2020 zip, loads 10 patients (~1–2 min first run),
then opens the interactive window.

**Controls**

| Input | Action |
|-------|--------|
| Left-click drag | Rotate |
| Middle-click drag / Shift+drag | Pan |
| Scroll wheel | Zoom |
| `N` / `→` | Next patient |
| `P` / `←` | Previous patient |
| `E` | Toggle edema (green) |
| `C` | Toggle necrotic core (yellow) |
| `T` | Toggle enhancing tumor (red) |
| `R` | Reset camera |
| `Q` | Quit |

### Generate the rotating GIF

```bash
python visualize_brats3d.py --gif
```

Renders 72 frames at 640×520 off-screen and saves `demo.gif` (~4 MB).

### Save static PNG screenshots for all 10 patients

The `brats3d_output/` folder already contains pre-rendered screenshots.
To regenerate:

```bash
python visualize_brats3d.py --screenshots
```

---

## Repository Structure

```
EEE515_medical_imaging/
├── visualize_brats3d.py      # Main pipeline (interactive + GIF + screenshots)
├── demo.gif                  # Animated 360-degree tumor rotation
├── README.md
├── brats3d_output/
│   ├── mosaic_all_patients.png
│   ├── patient_100_3d.png
│   ├── patient_101_3d.png
│   └── ...  (10 patients)
├── datasets.py
├── preprocess_brats.py
└── preprocess_monuseg.py
```

---

## Team Members

| Name | ASU ID | Role |
|------|--------|------|
| Luna | — | 3D visualization pipeline |
| Sarthak Bhat | — | Data preprocessing & segmentation |

---

## References

1. **BraTS 2020 Challenge**
   Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark
   (BRATS)," *IEEE Transactions on Medical Imaging*, 34(10), 2015.
   [https://doi.org/10.1109/TMI.2014.2377694](https://doi.org/10.1109/TMI.2014.2377694)

2. **Swin UNETR**
   Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation
   of Brain Tumors in MRI Images," *MICCAI BrainLes Workshop*, 2022.
   [https://arxiv.org/abs/2201.01266](https://arxiv.org/abs/2201.01266)

3. **MONAI Framework**
   MONAI Consortium, "Project MONAI," Zenodo, 2020.
   [https://doi.org/10.5281/zenodo.4323058](https://doi.org/10.5281/zenodo.4323058)

4. **Marching Cubes**
   Lorensen & Cline, "Marching Cubes: A High Resolution 3D Surface Construction
   Algorithm," *ACM SIGGRAPH*, 1987.
   [https://doi.org/10.1145/37401.37422](https://doi.org/10.1145/37401.37422)

5. **PyVista**
   Sullivan & Kaszynski, "PyVista: 3D plotting and mesh analysis through a
   streamlined interface for the Visualization Toolkit (VTK),"
   *Journal of Open Source Software*, 4(37), 2019.
   [https://doi.org/10.21105/joss.01450](https://doi.org/10.21105/joss.01450)

---

*Inspired by [neuro-voxel](https://github.com/asmarufoglu/neuro-voxel)*
