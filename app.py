# -*- coding: utf-8 -*-
"""
EEE-515 Medical Imaging -- BraTS2020 Streamlit Web App
=======================================================
Rotate the 3D tumor in your browser -- no download required.

Run:
    streamlit run app.py
"""

import io
import re
import zipfile

import h5py
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from skimage.measure import marching_cubes

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BraTS2020 3D Tumor Viewer",
    page_icon="🧠",
    layout="wide",
)

# ── constants ─────────────────────────────────────────────────────────────────
ZIP_PATH   = r"C:\Users\sbhat\Downloads\archive (3).zip"
N_PATIENTS = 10
SMOOTH_ITER = 15

REGIONS = [
    # label_val, display_name, plotly_color, opacity
    (4, "Enhancing Tumor (label 4)",     "rgb(230, 40,  40)",  0.90),
    (1, "Necrotic Core  (label 1)",      "rgb(255, 230, 25)",  0.80),
    (2, "Peritumoral Edema (label 2)",   "rgb(30,  210, 60)",  0.45),
]


# ── data helpers (cached so re-selecting a patient is instant) ────────────────
@st.cache_data(show_spinner=False)
def get_volume_ids():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        seen = {}
        for name in zf.namelist():
            m = re.search(r"volume_(\d+)_slice_", name)
            if m:
                seen[int(m.group(1))] = True
            if len(seen) >= N_PATIENTS:
                break
    return sorted(seen.keys())[:N_PATIENTS]


@st.cache_data(show_spinner=False)
def load_patient(vol_id: int):
    """
    Returns (meshes, voxel_counts).
    meshes: {region_name: (verts, faces)} -- raw marching-cubes output.
    """
    with zipfile.ZipFile(ZIP_PATH) as zf:
        pattern = re.compile(rf"volume_{vol_id}_slice_(\d+)\.h5$")
        slice_map = {}
        for name in zf.namelist():
            m = pattern.search(name)
            if m:
                slice_map[int(m.group(1))] = name

        if not slice_map:
            return None, None

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

    voxel_counts = {lv: int((vol == lv).sum()) for lv in [1, 2, 4]}

    meshes = {}
    for label_val, name, color, _ in REGIONS:
        binary = (vol == label_val).astype(np.float32)
        if binary.sum() < 200:
            continue
        try:
            verts, faces, _, _ = marching_cubes(binary, level=0.5)
            # Smooth by averaging neighbour positions (simple Laplacian)
            from scipy.ndimage import uniform_filter1d  # noqa: F401
            # skip heavy smooth for browser speed; marching cubes is smooth enough
            meshes[name] = (verts, faces, color)
        except Exception:
            continue

    return meshes, voxel_counts


def build_figure(meshes, voxel_counts, visible_flags):
    """Build a Plotly Figure with one Mesh3d trace per tumor region."""
    traces = []

    for label_val, name, color, opacity in REGIONS:
        if name not in meshes:
            continue
        if not visible_flags.get(name, True):
            continue

        verts, faces, _ = meshes[name]
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        traces.append(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=opacity,
            name=name,
            showlegend=True,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5,
            ),
            lightposition=dict(x=100, y=200, z=150),
        ))

    enh   = voxel_counts.get(4, 0)
    nec   = voxel_counts.get(1, 0)
    ede   = voxel_counts.get(2, 0)
    total = enh + nec + ede

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        margin=dict(l=0, r=0, t=40, b=0),
        height=680,
        title=dict(
            text=(
                f"<b>BraTS2020 — volume_{'{vol_id}'}</b>   "
                f"Total tumor: {total:,} mm\u00b3 ({total/1000:.1f} cm\u00b3)"
            ),
            font=dict(color="white", size=14),
            x=0.5,
        ),
        legend=dict(
            bgcolor="#222222",
            font=dict(color="white"),
            bordercolor="#444444",
            borderwidth=1,
        ),
        scene=dict(
            bgcolor="#111111",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       backgroundcolor="#111111"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       backgroundcolor="#111111"),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       backgroundcolor="#111111"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectmode="data",
        ),
        uirevision="tumor",   # keeps camera angle when traces update
    )
    return fig, enh, nec, ede, total


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 BraTS2020 Viewer")
    st.markdown("*EEE-515 Medical Imaging — ASU*")
    st.divider()

    vol_ids = get_volume_ids()
    selected_vol = st.selectbox(
        "Select patient",
        options=vol_ids,
        format_func=lambda v: f"volume_{v}",
    )

    st.divider()
    st.markdown("**Tumor regions**")
    show_enhancing = st.checkbox("Enhancing Tumor (red)",      value=True)
    show_necrotic  = st.checkbox("Necrotic Core (yellow)",     value=True)
    show_edema     = st.checkbox("Peritumoral Edema (green)",  value=True)

    st.divider()
    st.markdown(
        "**Controls**\n\n"
        "- 🖱️ Drag to rotate\n"
        "- 🖱️ Scroll to zoom\n"
        "- 🖱️ Right-drag to pan\n"
        "- Double-click to reset view"
    )
    st.divider()
    st.markdown(
        "**BraTS label map**\n\n"
        "🔴 Label 4 — Enhancing tumor\n\n"
        "🟡 Label 1 — Necrotic core\n\n"
        "🟢 Label 2 — Peritumoral edema"
    )


# ── main panel ────────────────────────────────────────────────────────────────
st.title("🧠 3D Brain Tumor Visualization")
st.caption(
    "Rotate the tumor in your browser using mouse drag. "
    "Use the sidebar to switch patients or toggle regions."
)

visible_flags = {
    "Enhancing Tumor (label 4)":   show_enhancing,
    "Necrotic Core  (label 1)":    show_necrotic,
    "Peritumoral Edema (label 2)": show_edema,
}

with st.spinner(f"Loading volume_{selected_vol} (~15 s first time) ..."):
    meshes, voxel_counts = load_patient(selected_vol)

if meshes is None:
    st.error(f"Could not load volume_{selected_vol}. Try another patient.")
    st.stop()

fig, enh, nec, ede, total = build_figure(meshes, voxel_counts, visible_flags)

# Fix title now that we have vol_id
fig.update_layout(title_text=(
    f"<b>BraTS2020 — volume_{selected_vol}</b>   "
    f"Total tumor: {total:,} mm\u00b3 ({total/1000:.1f} cm\u00b3)"
))

st.plotly_chart(fig, use_container_width=True)

# ── stats row ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Tumor Volume Measurements")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔴 Enhancing",    f"{enh:,} mm³",   f"{enh/1000:.2f} cm³")
with col2:
    st.metric("🟡 Necrotic Core", f"{nec:,} mm³",  f"{nec/1000:.2f} cm³")
with col3:
    st.metric("🟢 Edema",        f"{ede:,} mm³",   f"{ede/1000:.2f} cm³")
with col4:
    st.metric("Total Tumor",     f"{total:,} mm³", f"{total/1000:.2f} cm³")

st.divider()
st.caption(
    "Data: BraTS2020 (Kaggle) — ground-truth segmentation masks. "
    "Surface reconstruction via Marching Cubes (scikit-image). "
    "Voxel size: 1 mm isotropic (1 voxel = 1 mm³). "
    "Inspired by [neuro-voxel](https://github.com/asmarufoglu/neuro-voxel)."
)
