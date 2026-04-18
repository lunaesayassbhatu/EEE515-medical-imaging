"""
EEE-515 Medical Imaging -- BraTS2020 Interactive 3D Tumor Viewer
=================================================================
Run modes
---------
  Interactive viewer (default):
      python visualize_brats3d.py

  Generate rotating demo.gif:
      python visualize_brats3d.py --gif

Mouse controls (interactive mode)
----------------------------------
  Rotate        : Left-click + drag
  Pan           : Shift + left-click drag   OR  middle-click drag
  Zoom          : Scroll wheel
  Reset camera  : press R

Keyboard shortcuts
------------------
  N             : next patient
  P             : previous patient
  E             : toggle Edema (green)
  C             : toggle Necrotic core (yellow)
  T             : toggle Enhancing tumor (red)
  R             : reset camera to default view
  Q             : quit

BraTS label mapping
-------------------
  Ch 0  ->  Necrotic Core     (label 1)  ->  Yellow
  Ch 1  ->  Peritumoral Edema (label 2)  ->  Green
  Ch 2  ->  Enhancing Tumor   (label 4)  ->  Red
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
GIF_PATH    = r"C:\Users\sbhat\PycharmProjects\EEE515_medical_imaging\demo.gif"
SEG_RESULTS = r"C:\Users\sbhat\PycharmProjects\EEE515_medical_imaging\seg_results"
N_PATIENTS  = 10
SMOOTH_ITER = 15
MC_LEVEL    = 0.5
WINDOW_W    = 1300
WINDOW_H    = 860

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
    return sorted(seen.keys())[:n]


def load_volume_mask(zf, vol_id):
    """
    Stack all axial slices for vol_id into a (240, 240, 155) uint8 label volume.
    Priority: enhancing (4) > necrotic (1) > edema (2).
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
            mask = hf["mask"][:]
        lbl = np.zeros((240, 240), dtype=np.uint8)
        lbl[mask[:, :, 1] == 1] = 2
        lbl[mask[:, :, 0] == 1] = 1
        lbl[mask[:, :, 2] == 1] = 4
        vol[:, :, s_idx] = lbl
    return vol


def build_meshes(vol):
    """
    Run marching cubes on each tumor region.
    Returns dict: {region_name: pv.PolyData}
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
        n_f = faces.shape[0]
        face_arr = np.hstack([np.full((n_f, 1), 3), faces]).ravel()
        mesh = pv.PolyData(verts.astype(np.float32), face_arr).smooth(n_iter=SMOOTH_ITER)
        meshes[name] = mesh
    return meshes


def load_patient(zf, vol_id):
    """Load ground-truth masks from h5. Returns (meshes, voxel_counts) or (None, None)."""
    vol = load_volume_mask(zf, vol_id)
    if vol is None:
        return None, None
    voxel_counts = {lv: int((vol == lv).sum()) for lv in [1, 2, 4]}
    if sum(voxel_counts.values()) == 0:
        return None, None
    meshes = build_meshes(vol)
    if not meshes:
        return None, None
    return meshes, voxel_counts


def load_patient_predicted(vol_id):
    """
    Load AI-predicted label map from seg_results/pred_volume_<id>.npy.
    Returns (meshes, voxel_counts) or (None, None).
    """
    pred_path = os.path.join(SEG_RESULTS, f"pred_volume_{vol_id}.npy")
    if not os.path.exists(pred_path):
        print(f"  [!] No prediction found at {pred_path}")
        print(f"      Run: python segment_brats.py --n 10   to generate predictions first.")
        return None, None
    vol = np.load(pred_path)
    voxel_counts = {lv: int((vol == lv).sum()) for lv in [1, 2, 4]}
    if sum(voxel_counts.values()) == 0:
        print(f"  [!] Prediction for volume_{vol_id} has no tumor voxels")
        return None, None
    meshes = build_meshes(vol)
    if not meshes:
        return None, None
    return meshes, voxel_counts


# -----------------------------------------------------------------------------
# INTERACTIVE VIEWER
# -----------------------------------------------------------------------------
def run_interactive_viewer(zf, vol_ids, use_predicted=False):
    """
    Open a PyVista window immediately with the first patient.
    Patients are loaded on-demand when the user navigates with N / P.
    NO auto-rotation. Full mouse + keyboard control.

    use_predicted=True  -> load from seg_results/pred_volume_*.npy (AI output)
    use_predicted=False -> load from h5 ground-truth masks (default)
    """
    mode_label = "AI Predictions" if use_predicted else "Ground Truth"

    # ── shared mutable state (no class needed) ────────────────────────────────
    state = {
        "idx":    0,
        "vol_ids": vol_ids,
        "cache":  {},        # vol_id -> (meshes, voxel_counts)
        "vis":    {r[1]: True for r in REGIONS},
        "info_actor": None,
    }

    # ── plotter: explicitly interactive, trackball camera ─────────────────────
    pl = pv.Plotter(
        window_size=(WINDOW_W, WINDOW_H),
        title=f"BraTS2020 -- EEE-515  [{mode_label}]  |  "
              "Drag=Rotate  Scroll=Zoom  N/P=Patient  E/C/T=Toggle  Q=Quit",
        off_screen=False,     # always interactive
    )
    pl.set_background("#111111")
    pl.enable_anti_aliasing("ssaa")
    pl.enable_trackball_style()   # left-drag = rotate, NO auto-spin

    # ── helpers ───────────────────────────────────────────────────────────────
    def get_patient(idx):
        """Return (meshes, voxel_counts) for patients_idx, loading if needed."""
        vid = state["vol_ids"][idx]
        if vid not in state["cache"]:
            if use_predicted:
                print(f"  Loading AI prediction for volume_{vid} ...")
                meshes, vxc = load_patient_predicted(vid)
            else:
                print(f"  Loading volume_{vid} ...")
                meshes, vxc = load_patient(zf, vid)
            state["cache"][vid] = (meshes, vxc)
        return state["cache"][vid]

    def clear_scene():
        """Remove all tumor mesh actors from the renderer."""
        for name in [r[1] for r in REGIONS]:
            if name in pl.actors:
                pl.remove_actor(name, render=False)

    def draw_patient(idx):
        """Clear scene and draw the patient at `idx`."""
        clear_scene()
        meshes, vxc = get_patient(idx)
        if meshes is None:
            print(f"  [!] volume_{state['vol_ids'][idx]} has no valid meshes")
            return
        for label_val, name, color, opacity, _ in REGIONS:
            if name not in meshes:
                continue
            pl.add_mesh(
                meshes[name],
                color=color,
                opacity=opacity,
                smooth_shading=True,
                name=name,           # reuse same actor name -> clean replace
                reset_camera=False,
            )
            pl.actors[name].SetVisibility(state["vis"][name])

    def update_info():
        """Refresh the stats overlay in the top-right corner."""
        if state["info_actor"] is not None:
            try:
                pl.remove_actor(state["info_actor"], render=False)
            except Exception:
                pass
        idx  = state["idx"]
        vid  = state["vol_ids"][idx]
        _, vxc = state["cache"].get(vid, (None, None))
        if vxc is None:
            state["info_actor"] = None
            return
        enh   = vxc.get(4, 0)
        nec   = vxc.get(1, 0)
        ede   = vxc.get(2, 0)
        total = enh + nec + ede
        cm    = lambda v: v / 1000.0
        lines = [
            f"Patient {idx+1}/{len(vol_ids)}  vol={vid}",
            "",
            f"  [T] Enhancing  {enh:>7,} mm3  ({cm(enh):.2f} cm3)",
            f"  [C] Necrotic   {nec:>7,} mm3  ({cm(nec):.2f} cm3)",
            f"  [E] Edema      {ede:>7,} mm3  ({cm(ede):.2f} cm3)",
            f"  ---------------------",
            f"      Total      {total:>7,} mm3  ({cm(total):.2f} cm3)",
        ]
        state["info_actor"] = pl.add_text(
            "\n".join(lines),
            position="upper_right",
            font_size=9,
            color="white",
            font="courier",
        )

    def reset_camera():
        pl.camera_position = "iso"
        pl.reset_camera()
        pl.camera.zoom(1.2)

    # ── legend (static, keyboard-only, no widgets that eat mouse events) ──────
    legend_lines = [
        "  [E] Edema         (green)",
        "  [C] Necrotic Core (yellow)",
        "  [T] Enhancing     (red)",
        "",
        "  [N] next patient",
        "  [P] prev patient",
        "  [R] reset camera",
        "  [Q] quit",
    ]
    pl.add_text(
        "\n".join(legend_lines),
        position="lower_left",
        font_size=9,
        color="#cccccc",
        font="courier",
    )

    # ── key callbacks ─────────────────────────────────────────────────────────
    def nav(delta):
        state["idx"] = (state["idx"] + delta) % len(vol_ids)
        print(f"\nPatient {state['idx']+1}/{len(vol_ids)}: "
              f"volume_{vol_ids[state['idx']]}")
        draw_patient(state["idx"])
        update_info()
        reset_camera()
        pl.render()

    def make_toggle(region_name):
        def _toggle():
            state["vis"][region_name] = not state["vis"][region_name]
            flag = state["vis"][region_name]
            if region_name in pl.actors:
                pl.actors[region_name].SetVisibility(flag)
            status = "ON" if flag else "OFF"
            print(f"  {region_name}: {status}")
            pl.render()
        return _toggle

    pl.add_key_event("n",     lambda: nav(+1))
    pl.add_key_event("p",     lambda: nav(-1))
    pl.add_key_event("Right", lambda: nav(+1))
    pl.add_key_event("Left",  lambda: nav(-1))
    pl.add_key_event("r",     lambda: (reset_camera(), pl.render()))

    for _, name, _, _, key in REGIONS:
        pl.add_key_event(key, make_toggle(name))

    # ── load first patient and show window ────────────────────────────────────
    print(f"\nLoading patient 1/{len(vol_ids)}: volume_{vol_ids[0]} ...")
    draw_patient(0)
    update_info()
    reset_camera()

    print("\nWindow open. You are in full control.")
    print("  Drag with LEFT mouse button to rotate.")
    print("  Scroll to zoom. Press Q to quit.")
    pl.show()   # blocks here; returns when user closes window or presses Q


# -----------------------------------------------------------------------------
# GIF EXPORT  -- completely isolated from the interactive viewer
# -----------------------------------------------------------------------------
def make_demo_gif(zf, vol_ids):
    """
    Render a 360-degree orbit of patient 100 off-screen and save as demo.gif.
    Runs in a brand-new Plotter instance that is closed before returning,
    so it never interferes with the interactive viewer.
    """
    GIF_VOL_ID = vol_ids[0]   # first available volume (usually 100)
    GIF_FRAMES = 72
    GIF_W, GIF_H = 640, 520

    print(f"\nGIF mode: loading volume_{GIF_VOL_ID} ...")
    meshes, vxc = load_patient(zf, GIF_VOL_ID)
    if meshes is None:
        print("[ERROR] Could not build meshes. Exiting.")
        sys.exit(1)

    enh = vxc.get(4, 0)
    nec = vxc.get(1, 0)
    ede = vxc.get(2, 0)
    print(f"  Voxels - enh:{enh:,}  nec:{nec:,}  edema:{ede:,}")

    print(f"Rendering {GIF_FRAMES} frames off-screen ...")
    pl = pv.Plotter(off_screen=True, window_size=(GIF_W, GIF_H))
    pl.set_background("#0d0d0d")
    pl.enable_anti_aliasing("ssaa")

    for label_val, name, color, opacity, _ in REGIONS:
        if name in meshes:
            pl.add_mesh(meshes[name], color=color, opacity=opacity,
                        smooth_shading=True, name=name)

    pl.add_text(
        f"BraTS2020  volume_{GIF_VOL_ID}\n"
        f"Red=Enhancing({enh//1000}cm3)  "
        f"Yellow=Necrotic({nec//1000}cm3)  "
        f"Green=Edema({ede//1000}cm3)",
        position="upper_left", font_size=8, color="white", font="courier",
    )

    pl.camera_position = "iso"
    pl.reset_camera()
    pl.camera.zoom(1.2)

    path = pl.generate_orbital_path(n_points=GIF_FRAMES, shift=0.0,
                                    factor=2.4, viewup=(0, 0, 1))
    pl.open_gif(GIF_PATH)
    pl.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.05)
    pl.close()   # <-- always closed; never shown interactively

    size_mb = os.path.getsize(GIF_PATH) / 1e6
    print(f"Saved: {GIF_PATH}  ({size_mb:.1f} MB)")


# -----------------------------------------------------------------------------
# 10-PATIENT COMPARISON GRID  (python visualize_brats3d.py --comparison)
# -----------------------------------------------------------------------------
COMPARISON_PATH = (
    r"C:\Users\sbhat\PycharmProjects\EEE515_medical_imaging"
    r"\all_10_patients_comparison.png"
)
THUMB_W, THUMB_H = 420, 340   # pixels per patient thumbnail


def _render_patient_thumbnail(meshes, voxel_counts):
    """Render one patient off-screen and return an RGBA numpy array."""
    pl = pv.Plotter(off_screen=True, window_size=(THUMB_W, THUMB_H))
    pl.set_background("#0d0d0d")
    pl.enable_anti_aliasing("ssaa")

    for label_val, name, color, opacity, _ in REGIONS:
        if name in meshes:
            pl.add_mesh(meshes[name], color=color, opacity=opacity,
                        smooth_shading=True, name=name)

    pl.camera_position = "iso"
    pl.reset_camera()
    pl.camera.zoom(1.25)
    pl.add_light(pv.Light(position=(200, 200, 400), intensity=0.6))

    img = pl.screenshot(return_img=True)   # (H, W, 3) uint8
    pl.close()
    return img


def make_comparison_grid(zf, vol_ids):
    """
    Render all 10 patients off-screen, tile into a 2x5 grid,
    annotate each panel with patient ID and tumor volumes, save PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    COLS, ROWS = 5, 2
    assert len(vol_ids) <= COLS * ROWS

    # ── per-patient data ──────────────────────────────────────────────────────
    panels = []   # list of (vol_id, img_array, voxel_counts)

    for i, vid in enumerate(vol_ids, 1):
        print(f"  [{i}/{len(vol_ids)}] Rendering volume_{vid} ...")
        meshes, vxc = load_patient(zf, vid)
        if meshes is None:
            # blank black panel
            img = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
            vxc = {1: 0, 2: 0, 4: 0}
        else:
            img = _render_patient_thumbnail(meshes, vxc)
        panels.append((vid, img, vxc))

    # ── figure layout ─────────────────────────────────────────────────────────
    FIG_W = COLS * (THUMB_W / 100) + 0.4   # inches
    # Extra vertical space per row: title + 3 stat lines + spacing
    PANEL_H_IN = THUMB_H / 100
    LABEL_H_IN = 1.05     # inches for text block below each thumbnail
    FIG_H = ROWS * (PANEL_H_IN + LABEL_H_IN) + 1.6   # + header

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#0a0a0a")

    # Header
    fig.text(
        0.5, 1 - 0.18 / FIG_H,
        "BraTS2020  --  3D Glioma Segmentation  --  10 Patient Comparison",
        ha="center", va="top", fontsize=15, fontweight="bold",
        color="white", fontfamily="monospace",
    )
    fig.text(
        0.5, 1 - 0.72 / FIG_H,
        "Red = Enhancing Tumor (ET)     Yellow = Necrotic Core (NC)"
        "     Green = Peritumoral Edema (ED)",
        ha="center", va="top", fontsize=9, color="#bbbbbb",
        fontfamily="monospace",
    )

    # Legend patches (top-right)
    legend_handles = [
        mpatches.Patch(color="#E5191A", label="Enhancing (ET)"),
        mpatches.Patch(color="#FFE61A", label="Necrotic (NC)"),
        mpatches.Patch(color="#18D62A", label="Edema (ED)"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper right",
        fontsize=8, framealpha=0.15, labelcolor="white",
        facecolor="#222222", edgecolor="#444444",
    )

    # Grid spec: each row has image + text block
    # heights_ratios: [img_row, text_row, img_row, text_row]
    h_ratios = []
    for _ in range(ROWS):
        h_ratios += [PANEL_H_IN, LABEL_H_IN]
    gs = GridSpec(
        ROWS * 2, COLS,
        figure=fig,
        hspace=0.02,
        wspace=0.04,
        top=1 - 1.1 / FIG_H,
        bottom=0.02,
        left=0.02,
        right=0.98,
        height_ratios=h_ratios,
    )

    for idx, (vid, img, vxc) in enumerate(panels):
        row = idx // COLS
        col = idx % COLS
        gs_img  = gs[row * 2,     col]
        gs_text = gs[row * 2 + 1, col]

        # ── thumbnail ─────────────────────────────────────────────────────────
        ax_img = fig.add_subplot(gs_img)
        ax_img.imshow(img)
        ax_img.set_facecolor("#0d0d0d")
        ax_img.axis("off")

        # Thin coloured border
        for spine in ax_img.spines.values():
            spine.set_edgecolor("#333333")
            spine.set_linewidth(1.5)

        # ── stats block ───────────────────────────────────────────────────────
        ax_txt = fig.add_subplot(gs_text)
        ax_txt.set_facecolor("#0d0d0d")
        ax_txt.axis("off")

        enh   = vxc.get(4, 0)
        nec   = vxc.get(1, 0)
        ede   = vxc.get(2, 0)
        total = enh + nec + ede

        # Patient ID header
        ax_txt.text(
            0.5, 0.97, f"Volume {vid}",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="white", fontfamily="monospace",
            transform=ax_txt.transAxes,
        )
        # Volume stats
        stats = [
            (f"ET  {enh:>7,} mm3  ({enh/1000:5.1f} cm3)", "#E5191A"),
            (f"NC  {nec:>7,} mm3  ({nec/1000:5.1f} cm3)", "#FFE61A"),
            (f"ED  {ede:>7,} mm3  ({ede/1000:5.1f} cm3)", "#18D62A"),
            (f"TOT {total:>7,} mm3  ({total/1000:5.1f} cm3)", "#cccccc"),
        ]
        y = 0.72
        for text, color in stats:
            ax_txt.text(
                0.5, y, text,
                ha="center", va="top", fontsize=7.2,
                color=color, fontfamily="monospace",
                transform=ax_txt.transAxes,
            )
            y -= 0.21

    fig.savefig(
        COMPARISON_PATH, dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    size_mb = os.path.getsize(COMPARISON_PATH) / 1e6
    print(f"\nSaved: {COMPARISON_PATH}  ({size_mb:.1f} MB)")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    gif_mode        = "--gif"        in sys.argv
    predicted_mode  = "--predicted"  in sys.argv
    comparison_mode = "--comparison" in sys.argv

    print("=" * 60)
    if gif_mode:
        print("BraTS2020 -- GIF export mode")
    elif predicted_mode:
        print("BraTS2020 -- EEE-515 Viewer  [AI Predictions]")
    elif comparison_mode:
        print("BraTS2020 -- 10-Patient Comparison Grid")
    else:
        print("BraTS2020 -- EEE-515 Interactive 3D Viewer  [Ground Truth]")
    print("=" * 60)

    print(f"\nOpening zip: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        vol_ids = get_volume_ids(zf, n=N_PATIENTS)
        print(f"Found {len(vol_ids)} volumes: {vol_ids}")

        if gif_mode:
            make_demo_gif(zf, vol_ids)
        elif comparison_mode:
            make_comparison_grid(zf, vol_ids)
        else:
            run_interactive_viewer(zf, vol_ids, use_predicted=predicted_mode)


if __name__ == "__main__":
    main()
