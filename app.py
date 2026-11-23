"""Brain MRI Explorer: interactive medical-image visualization dashboard."""
from __future__ import annotations

import streamlit as st

from mri_utils import (
    as_rgb,
    csv_bytes,
    load_volume,
    make_demo,
    png_bytes,
    region_summary,
    take_slice,
)

st.set_page_config(
    page_title="Brain MRI Explorer | MRI Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; max-width: 1300px;}
      .hero {padding: 18px 22px; background: linear-gradient(120deg, #112a42, #145b66);
             color: #f5fbff; border-radius: 16px; margin-bottom: 20px;}
      .hero h1 {color: #f5fbff; margin: 0; font-weight: 720;}
      .hero p {margin-bottom: 0; color: #daedf5;}
      .disclaimer {font-size: .92rem; color: #b7c5d4;}
    </style>
    <div class='hero'>
      <h1>🧠 Brain MRI Explorer</h1>
      <p>Brain tumor research companion: volumetric MRI viewing and segmentation-label analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "RESEARCH DEMONSTRATION ONLY. This dashboard does not detect tumors or "
    "predict survival. Do not upload identifiable patient data or use it for medical decisions."
)

with st.sidebar:
    st.header("🔬 Select data")
    source = st.radio("MRI source", ["Built-in synthetic sample", "Upload my own data"])
    st.caption("The built-in sample is artificially generated, not a real patient scan.")
    uploaded_mri = None
    uploaded_mask = None
    if source == "Upload my own data":
        uploaded_mri = st.file_uploader(
            "MRI volume (.nii, .nii.gz, .npy)", type=["nii", "gz", "npy"],
            help="A 3D volume, or a 4D array with modalities on its final axis.",
        )
        uploaded_mask = st.file_uploader(
            "Optional labeled segmentation mask", type=["nii", "gz", "npy"],
            help="Must be a 3D array with the same spatial dimensions as the MRI.",
        )
        st.info("Upload only publicly shareable, de-identified research data.")

if source == "Built-in synthetic sample":
    volume, mask, spacing = make_demo()
    source_label = "artificial example"
elif uploaded_mri is None:
    st.info("👈 Use the sidebar to upload an MRI volume, or select the built-in synthetic sample.")
    st.stop()
else:
    try:
        volume, spacing = load_volume(uploaded_mri.getvalue(), uploaded_mri.name)
        source_label = "uploaded research scan"
        mask = None
        if uploaded_mask is not None:
            mask, _ = load_volume(uploaded_mask.getvalue(), uploaded_mask.name)
            if mask.ndim != 3:
                raise ValueError("Segmentation masks must be 3-D.")
            if tuple(mask.shape) != tuple(volume.shape[:3]):
                raise ValueError("Mask spatial dimensions do not match the MRI volume.")
    except Exception as exc:
        st.error(f"Could not read the supplied file: {exc}")
        st.stop()

if volume.ndim == 4:
    channels = int(volume.shape[3])
    default_names = ["FLAIR", "T1ce", "T2", "T1"] if channels in (3, 4) else []
    # For a three-channel BraTS preprocessed image, expected order is FLAIR, T1ce, T2.
    names = default_names[:channels] if default_names else [f"Channel {i+1}" for i in range(channels)]
    selection = st.selectbox("Displayed MRI channel (verify your channel ordering)", names)
    channel_index = names.index(selection)
    display_vol = volume[:, :, :, channel_index]
else:
    display_vol = volume
    st.caption("Displaying one three-dimensional MRI channel.")

st.subheader("1. Explore MRI slices")
ctrl1, ctrl2 = st.columns([1, 2])
with ctrl1:
    plane = st.radio("View orientation", ["Axial", "Coronal", "Sagittal"], horizontal=False)
with ctrl2:
    axis_no = {"Axial": 2, "Coronal": 1, "Sagittal": 0}[plane]
    number = st.slider("Slice", 0, int(display_vol.shape[axis_no] - 1), int(display_vol.shape[axis_no] // 2))

img_slice = take_slice(display_vol, plane, number)
mask_slice = take_slice(mask, plane, number) if mask is not None else None
img_rgb = as_rgb(img_slice)
overlay_rgb = as_rgb(img_slice, mask_slice) if mask_slice is not None else None

left, right = st.columns(2)
with left:
    st.markdown("**MRI view**")
    st.image(img_rgb, use_container_width=True, clamp=True)
    st.download_button("⬇️ Save MRI slice (PNG)", png_bytes(img_rgb), "mri_slice.png", "image/png")
with right:
    if overlay_rgb is not None:
        st.markdown("**Provided / synthetic mask overlay**")
        st.image(overlay_rgb, use_container_width=True, clamp=True)
        st.download_button("⬇️ Save overlay (PNG)", png_bytes(overlay_rgb), "mask_overlay.png", "image/png")
    else:
        st.info("Add an existing segmentation mask in the sidebar to see an overlay here. "
                "This app will not invent tumor boundaries.")

st.subheader("2. Volume information")
m1, m2, m3 = st.columns(3)
m1.metric("MRI spatial shape", " × ".join(str(int(s)) for s in volume.shape[:3]))
m2.metric("Input type", "4D multichannel" if volume.ndim == 4 else "3D single-channel")
m3.metric("Mask available", "Yes" if mask is not None else "No")

if source == "Upload my own data" and uploaded_mri is not None and uploaded_mri.name.lower().endswith(".npy"):
    st.caption("NumPy .npy files do not carry anatomical spacing; change these values if known.")
    a, b, c = st.columns(3)
    spacing = (
        a.number_input("X spacing (mm)", min_value=0.01, max_value=20.0, value=1.0),
        b.number_input("Y spacing (mm)", min_value=0.01, max_value=20.0, value=1.0),
        c.number_input("Z spacing (mm)", min_value=0.01, max_value=20.0, value=1.0),
    )
else:
    st.caption(f"Voxel spacing: {spacing[0]:g} × {spacing[1]:g} × {spacing[2]:g} mm")

if mask is not None:
    stats = region_summary(mask, spacing)
    a, b, c = st.columns(3)
    a.metric("Annotated voxels", f"{stats['annotated_voxels']:,}")
    b.metric("Annotated fraction", f"{stats['annotated_percent']:.2f}%")
    c.metric("Annotated region volume", f"{stats['annotated_volume_ml']:.2f} mL")
    st.caption("These describe an existing annotation, NOT an AI-generated tumor prediction. "
               "For .npy data, the mL estimate depends on your spacing inputs.")
    st.download_button(
        "⬇️ Save region summary (CSV)", csv_bytes(stats, source_label),
        "mri_region_summary.csv", "text/csv",
    )
else:
    st.info("Upload a valid mask to unlock label-area and volume summaries.")

with st.expander("3. Academic research context and implementation scope"):
    st.markdown(
        """
        **Academic project:** Brain Tumor Detection and Survival Prediction Using Deep
        Learning Techniques (2021–2025).  
        **Project team:** B. Sai Kiran, S. Yoshitha, D. Asharitha, S. Yoshritha.

        The research describes BraTS 2020 MRI preprocessing, V-Net segmentation,
        image-derived features and survival modeling.

        **This dashboard:** NIfTI/NumPy volume inspection, image-plane navigation,
        visualization of existing label masks, region measurements and image/CSV exports.

        **Research-model inference:** The trained segmentation and survival models are
        not deployed here. This viewer does not generate diagnostic or prognostic results.
        """
    )

st.caption("Interactive research visualization. Uploaded files are processed by the hosting service; "
           "do not submit private medical records or identifiable patient information.")
