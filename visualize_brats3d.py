"""
EEE-515 Medical Imaging -- BraTS2020 Interactive 3D Tumor Viewer
===============================================================
Reconstructs 3D tumor regions from ground-truth segmentation masks
stored as 2D h5 slices in the Kaggle BraTS2020 dataset.

Controls
--------
  Rotate          : Left-click drag
  Pan             : Middle-click drag  (or Shift + Left-click)
  Zoom            : Scroll wheel
  Reset view      : R key
  Next patient    : N key  (or Right arrow)
  Prev patient    : P key  (or Left arrow)
  Toggle Edema    : E key
  Toggle Necrotic : C key
  Toggle Enhancing: T key
  Quit            : Q key

BraTS label mapping
-------------------
  Ch 0 -> Necrotic Core    (label 1) -> Yellow
  Ch 1 -> Peritumoral Edema(label 2) -> Green
  Ch 2 -> Enhancing Tumor  (label 4) -> Red
"""

import os
import re
import io
import sys
import zipfile

import numpy as np
import h5py
from skimage.measure import marching_cubes
import pyvista as pv

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
ZIP_PATH    = r"C:\Users\sbhat\Downloads\archive (3).zip"
OUT_DIR     = r"C:\Users\sbhat\PycharmProjects\EEE515_medical_imaging\brats3d_output"
N_PATIENTS  = 10
SMOOTH_ITER = 15
MC_LEVEL    = 0.5
WINDOW_W    = 1400
WINDOW_H    = 900

# BraTS voxel size: 1 mm isotropic -> each voxel = 1 mm3
VOXEL_MM3 = 1.0

# Region definition: (label_value, display_name, hex_color, opacity, toggle_key)
REGIONS = [
    (2, "Edema",         "#18D62A", 0.40, "e"),
    (1, "Necrotic Core", "#FFE61A", 0.75, "c"),
    (4, "Enhancing",     "#E5191A", 0.90, "t"),
]

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
def get_volume_ids(zf, n=10):
    """Return the first n unique integer volume IDs found in the zip."""
    seen = {}
    for name in zf.namelist():
        m = re.search(r"volume_(\d+)_slice_", name)
        if m:
            seen[int(m.group(1))] = True
        if len(seen) >= n:
            break
    ids = sorted(seen.keys())[:n]
    print(f"  Found volumes: {ids}")
    return ids


def load_volume_mask(zf, vol_id):
    """
    Stack all axial slices for vol_id into a (240, 240, 155) uint8 label volume.

    Label priority (highest wins per voxel):
      4 = enhancing tumor  (ch 2)
      1 = necrotic core    (ch 0)
      2 = edema            (ch 1)
    """
    pattern = re.compile(rf"volume_{vol_id}_slice_(\d+)\.h5$")
    slice_map = {}
    for name in zf.namelist():
        m = pattern.search(name)
        if m:
            slice_map[int(m.group(1))] = name

    if not slice_map:
        return None

    n_slices = max(slice_map.keys()) + 1
    vol = np.zeros((240, 240, n_slices), dtype=np.uint8)

    for s_idx in sorted(slice_map.keys()):
        with zf.open(slice_map[s_idx]) as raw:
            buf = io.BytesIO(raw.read())
        with h5py.File(buf, "r") as hf:
            mask = hf["mask"][:]          # (240, 240, 3) binary uint8

        label_slice = np.zeros((240, 240), dtype=np.uint8)
        label_slice[mask[:, :, 1] == 1] = 2   # edema
        label_slice[mask[:, :, 0] == 1] = 1   # necrotic (overwrites edema)
        label_slice[mask[:, :, 2] == 1] = 4   # enhancing (highest priority)
        vol[:, :, s_idx] = label_slice

    return vol


def build_meshes(vol):
    """
    Run marching cubes on each tumor class binary sub-volume.
    Returns dict: {region_name: (pv.PolyData, hex_color)}
    """
    meshes = {}
    for label_val, name, color, _, _ in REGIONS:
        binary = (vol == label_val).astype(np.float32)

        if binary.sum() < 200:
            print(f"    Skipping {name}: only {int(binary.sum())} voxels")
            continue

        try:
            verts, faces, _, _ = marching_cubes(binary, level=MC_LEVEL)
        except Exception as exc:
            print(f"    marching_cubes failed for {name}: {exc}")
            continue

        n_faces = faces.shape[0]
        face_arr = np.hstack([np.full((n_faces, 1), 3), faces]).ravel()
        mesh = pv.PolyData(verts.astype(np.float32), face_arr)
        mesh = mesh.smooth(n_iter=SMOOTH_ITER)
        meshes[name] = (mesh, color)

    return meshes


# -----------------------------------------------------------------------------
# INTERACTIVE VIEWER
# -----------------------------------------------------------------------------
class BraTSViewer:
    """
    Single PyVista window showing all 10 patients.
    Patients are pre-loaded; navigation switches actor visibility.
    """

    def __init__(self, patients_data):
        # patients_data: list of (vol_id, meshes_dict, voxel_counts_dict)
        self.patients  = patients_data
        self.n         = len(patients_data)
        self.idx       = 0                              # current patient index
        self.vis       = {r[1]: True for r in REGIONS} # per-region toggle state

        # all_actors[vol_id][region_name] = vtk actor
        self.all_actors: dict = {}
        self.info_actor = None
        self.pl: pv.Plotter = None

    # -- public entry point ----------------------------------------------------
    def run(self):
        self.pl = pv.Plotter(
            window_size=(WINDOW_W, WINDOW_H),
            title="BraTS2020 -- EEE-515 Interactive Tumor Viewer",
        )
        self.pl.set_background("#0d0d0d")
        self.pl.enable_anti_aliasing()

        # Pre-load every patient's meshes (all hidden except patient 0)
        print("\nBuilding scene (all patients loaded once) ...")
        for i, (vol_id, meshes, voxel_counts) in enumerate(self.patients):
            self.all_actors[vol_id] = {}
            first = (i == 0)
            for label_val, name, color, opacity, _ in REGIONS:
                if name not in meshes:
                    continue
                mesh, _ = meshes[name]
                actor_key = f"{vol_id}_{name}"
                self.pl.add_mesh(
                    mesh,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                    name=actor_key,
                )
                actor = self.pl.actors[actor_key]
                actor.SetVisibility(first and self.vis[name])
                self.all_actors[vol_id][name] = actor
            print(f"  [{i+1}/{self.n}] volume_{vol_id} added to scene")

        # Key bindings for navigation
        self.pl.add_key_event("n",     lambda: self._navigate(+1))
        self.pl.add_key_event("p",     lambda: self._navigate(-1))
        self.pl.add_key_event("Right", lambda: self._navigate(+1))
        self.pl.add_key_event("Left",  lambda: self._navigate(-1))
        self.pl.add_key_event("r",     self._reset_camera)

        # Key bindings for region toggles
        for _, name, _, _, key in REGIONS:
            self.pl.add_key_event(key, self._make_key_toggle(name))

        # Checkbox widgets + region labels (left sidebar)
        self._add_checkbox_widgets()

        # Instructions overlay (bottom left)
        self.pl.add_text(
            "N / -> : next patient      P / ? : prev patient\n"
            "E: edema   C: necrotic   T: enhancing   R: reset   Q: quit",
            position="lower_left",
            font_size=9,
            color="#aaaaaa",
            font="courier",
        )

        # Info panel (upper right) for current patient stats
        self._update_info()

        # Initial camera
        self.pl.camera_position = "iso"
        self.pl.reset_camera()
        self.pl.camera.zoom(1.15)

        self.pl.show()

    # -- navigation ------------------------------------------------------------
    def _navigate(self, delta):
        old_id = self.patients[self.idx][0]
        self.idx = (self.idx + delta) % self.n
        new_id = self.patients[self.idx][0]

        # Hide old patient
        for actor in self.all_actors.get(old_id, {}).values():
            actor.SetVisibility(False)

        # Show new patient (respecting toggle state)
        for name, actor in self.all_actors.get(new_id, {}).items():
            actor.SetVisibility(self.vis.get(name, True))

        self._update_info()
        self.pl.reset_camera()
        self.pl.render()

    def _reset_camera(self):
        self.pl.camera_position = "iso"
        self.pl.reset_camera()
        self.pl.camera.zoom(1.15)
        self.pl.render()

    # -- toggle helpers --------------------------------------------------------
    def _make_checkbox_callback(self, name):
        """Returns a checkbox callback that toggles region `name`."""
        def callback(flag):
            self.vis[name] = flag
            vid = self.patients[self.idx][0]
            actor = self.all_actors.get(vid, {}).get(name)
            if actor:
                actor.SetVisibility(flag)
            self.pl.render()
        return callback

    def _make_key_toggle(self, name):
        """Returns a key callback that flips the toggle state of region `name`."""
        def callback():
            new_state = not self.vis[name]
            self.vis[name] = new_state
            vid = self.patients[self.idx][0]
            actor = self.all_actors.get(vid, {}).get(name)
            if actor:
                actor.SetVisibility(new_state)
            # Sync checkbox widget state by re-rendering
            self.pl.render()
        return callback

    # -- UI widgets ------------------------------------------------------------
    def _add_checkbox_widgets(self):
        """Add three stacked checkbox buttons in the top-left corner."""
        # Widget layout: start near top, step downward
        # PyVista widget position origin is bottom-left of window
        x_box  = 15
        x_text = 55
        box_h  = 35   # widget size in pixels
        gap    = 12   # gap between widgets
        step   = box_h + gap

        # Start position from the top
        y_start = WINDOW_H - 80   # pixels from bottom

        for i, (label_val, name, color, opacity, key) in enumerate(REGIONS):
            y = y_start - i * step
            self.pl.add_checkbox_button_widget(
                self._make_checkbox_callback(name),
                value=True,
                position=(x_box, y),
                size=box_h,
                border_size=2,
                color_on=color,
                color_off="#444444",
                background_color="#1a1a1a",
            )
            self.pl.add_text(
                f"{name}  [{key}]",
                position=(x_text, y + 8),
                font_size=10,
                color=color,
                font="courier",
            )

    # -- info panel ------------------------------------------------------------
    def _update_info(self):
        """Refresh the stats panel for the current patient."""
        if self.info_actor is not None:
            try:
                self.pl.remove_actor(self.info_actor)
            except Exception:
                pass

        vol_id, _, voxel_counts = self.patients[self.idx]

        enh = voxel_counts.get(4, 0)
        nec = voxel_counts.get(1, 0)
        ede = voxel_counts.get(2, 0)
        total = enh + nec + ede

        def mm3_to_cm3(v):
            return v / 1000.0

        lines = [
            "=== Patient Info ========",
            f"  volume_{vol_id}",
            f"  [{self.idx + 1} / {self.n}]",
            "",
            "=== Tumor Volumes =======",
            f"  Enhancing : {enh:>7,} mm3  ({mm3_to_cm3(enh):.2f} cm3)",
            f"  Necrotic  : {nec:>7,} mm3  ({mm3_to_cm3(nec):.2f} cm3)",
            f"  Edema     : {ede:>7,} mm3  ({mm3_to_cm3(ede):.2f} cm3)",
            f"  -------------------------",
            f"  Total     : {total:>7,} mm3  ({mm3_to_cm3(total):.2f} cm3)",
        ]

        self.info_actor = self.pl.add_text(
            "\n".join(lines),
            position="upper_right",
            font_size=9,
            color="#dddddd",
            font="courier",
        )


# -----------------------------------------------------------------------------
# GIF EXPORT  (python visualize_brats3d.py --gif)
# -----------------------------------------------------------------------------
GIF_VOL_ID  = 100   # patient to feature in the demo GIF
GIF_FRAMES  = 72    # one frame every 5 degrees -> smooth 360 rotation
GIF_PATH    = r"C:\Users\sbhat\PycharmProjects\EEE515_medical_imaging\demo.gif"
GIF_W, GIF_H = 640, 520


def make_demo_gif():
    """
    Render a 360-degree orbit of patient GIF_VOL_ID off-screen and save
    as an animated GIF suitable for embedding in the GitHub README.
    """
    print("=" * 60)
    print("BraTS2020 -- Generating demo.gif")
    print("=" * 60)

    print(f"\nLoading volume {GIF_VOL_ID} from zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        vol = load_volume_mask(zf, GIF_VOL_ID)

    if vol is None:
        print("[ERROR] Could not load volume. Exiting.")
        sys.exit(1)

    voxel_counts = {lv: int((vol == lv).sum()) for lv in [1, 2, 4]}
    total = sum(voxel_counts.values())
    print(f"  Voxels - enhancing:{voxel_counts[4]:,}  "
          f"necrotic:{voxel_counts[1]:,}  "
          f"edema:{voxel_counts[2]:,}  total:{total:,}")

    print("Building meshes...")
    meshes = build_meshes(vol)
    if not meshes:
        print("[ERROR] No meshes built. Exiting.")
        sys.exit(1)

    print(f"Rendering {GIF_FRAMES} frames ({360//GIF_FRAMES} deg/frame)...")
    pl = pv.Plotter(off_screen=True, window_size=(GIF_W, GIF_H))
    pl.set_background("#0d0d0d")
    pl.enable_anti_aliasing()

    # Add all three tumor regions
    for label_val, name, color, opacity, _ in REGIONS:
        if name in meshes:
            mesh, _ = meshes[name]
            pl.add_mesh(mesh, color=color, opacity=opacity,
                        smooth_shading=True, name=name)

    # Title text burned into every frame
    enh = voxel_counts.get(4, 0)
    nec = voxel_counts.get(1, 0)
    ede = voxel_counts.get(2, 0)
    pl.add_text(
        f"BraTS2020 -- volume_{GIF_VOL_ID}\n"
        f"Red=Enhancing({enh//1000}cm3)  "
        f"Yellow=Necrotic({nec//1000}cm3)  "
        f"Green=Edema({ede//1000}cm3)",
        position="upper_left",
        font_size=9,
        color="#dddddd",
        font="courier",
    )

    # Generate smooth orbital camera path and write GIF
    pl.camera_position = "iso"
    pl.reset_camera()
    pl.camera.zoom(1.2)

    path = pl.generate_orbital_path(
        n_points=GIF_FRAMES,
        shift=0.0,
        factor=2.4,
        viewup=(0, 0, 1),
    )

    pl.open_gif(GIF_PATH)
    pl.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.05)
    pl.close()

    size_mb = os.path.getsize(GIF_PATH) / 1_000_000
    print(f"\nSaved: {GIF_PATH}  ({size_mb:.1f} MB)")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    # --gif mode: generate rotating GIF and exit
    if "--gif" in sys.argv:
        make_demo_gif()
        return

    print("=" * 60)
    print("BraTS2020 -- EEE-515 Interactive 3D Viewer")
    print("=" * 60)

    print(f"\nOpening zip: {ZIP_PATH}")
    patients_data = []

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        vol_ids = get_volume_ids(zf, n=N_PATIENTS)

        for i, vid in enumerate(vol_ids, 1):
            print(f"\n[{i}/{N_PATIENTS}] Loading volume {vid} ...")

            vol = load_volume_mask(zf, vid)
            if vol is None:
                print(f"  [!] No slices found - skipping")
                continue

            voxel_counts = {lv: int((vol == lv).sum()) for lv in [1, 2, 4]}
            total = sum(voxel_counts.values())
            print(f"  Voxels - enhancing:{voxel_counts[4]:,}  "
                  f"necrotic:{voxel_counts[1]:,}  "
                  f"edema:{voxel_counts[2]:,}  "
                  f"total:{total:,}")

            if total == 0:
                print("  [!] No tumor mask -- skipping")
                continue

            meshes = build_meshes(vol)
            if not meshes:
                print("  [!] No meshes built -- skipping")
                continue

            patients_data.append((vid, meshes, voxel_counts))

    if not patients_data:
        print("\n[ERROR] No patients could be loaded. Exiting.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Loaded {len(patients_data)} patients. Opening interactive viewer...")
    print("Use N / P keys or arrow keys to navigate between patients.")
    print("Use E / C / T keys to toggle tumor regions.")
    print("Press Q to quit.")
    print("=" * 60)

    viewer = BraTSViewer(patients_data)
    viewer.run()


if __name__ == "__main__":
    main()
