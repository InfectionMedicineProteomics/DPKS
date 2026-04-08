"""
Page 3 — Normalization & Scaling
Normalise sample intensities and optionally scale at the feature level.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import streamlit as st
from utils.state import render_sidebar, require_step, get_qm, set_qm
from utils.plots import intensity_boxplot

st.set_page_config(page_title="3. Normalization & Scaling — DPKS GUI", layout="wide")
render_sidebar()

st.title("3. Normalization & Scaling")
st.markdown(
    "Normalise intensities across samples to remove systematic biases, "
    "then optionally scale at the feature (protein/precursor) level."
)

if not require_step("qm_filtered", "Filtering", "2. Filtering"):
    st.stop()

qm_input = get_qm("qm_filtered")

st.divider()

# ── Normalization ──────────────────────────────────────────────────────────
st.subheader("📐 Sample Normalization")

col1, col2 = st.columns(2)

with col1:
    norm_method = st.selectbox(
        "Normalization method",
        options=["mean", "median", "tic", "log2"],
        help=(
            "**mean** — divide each sample by its mean intensity (recommended). "
            "**median** — divide by median. "
            "**tic** — total ion current. "
            "**log2** — log₂ transform only (no normalisation)."
        ),
    )

    log_transform = st.checkbox(
        "Apply log₂ transform after normalization",
        value=True,
        help="Not applied if method is already 'log2'.",
        disabled=(norm_method == "log2"),
    )

with col2:
    has_rt = "RT" in qm_input.row_annotations.columns or \
             "RetentionTime" in qm_input.row_annotations.columns

    use_rt_window = st.checkbox(
        "Use RT sliding-window filter",
        value=False,
        disabled=not has_rt,
        help="Normalise within retention-time windows. Requires an RT column in the data.",
    )

    if not has_rt:
        st.caption("⚠️ No RT column detected — sliding window filter not available.")

if use_rt_window:
    st.markdown("**RT sliding-window parameters**")
    rw1, rw2, rw3, rw4 = st.columns(4)
    minimum_data_points = rw1.number_input("Min. data points", min_value=10, value=100, step=10)
    stride = rw2.number_input("Stride", min_value=1, value=1, step=1)
    use_overlapping = rw3.checkbox("Overlapping windows", value=True)
    rt_unit = rw4.selectbox("RT unit", options=["minute", "second"])

st.divider()

# ── Scaling ────────────────────────────────────────────────────────────────
st.subheader("📏 Feature-level Scaling (optional)")

apply_scaling = st.checkbox(
    "Apply feature scaling after normalization",
    value=False,
    help="Scales each feature (row) independently. Useful before ML steps.",
)

scale_method = None
if apply_scaling:
    scale_method = st.selectbox(
        "Scaling method",
        options=["zscore", "minmax", "absmax"],
        help=(
            "**zscore** — standardise to zero mean and unit variance. "
            "**minmax** — scale to [0, 1] range. "
            "**absmax** — scale by the absolute maximum."
        ),
    )

st.divider()

# ── Apply button ───────────────────────────────────────────────────────────
if st.button("▶️ Apply Normalization & Scaling", type="primary"):
    try:
        with st.spinner("Normalising…"):
            qm_norm = copy.deepcopy(qm_input)

            norm_kwargs = dict(
                method=norm_method,
                log_transform=log_transform if norm_method != "log2" else False,
            )

            if use_rt_window:
                norm_kwargs.update(
                    use_rt_sliding_window_filter=True,
                    minimum_data_points=int(minimum_data_points),
                    stride=int(stride),
                    use_overlapping_windows=use_overlapping,
                    rt_unit=rt_unit,
                )

            qm_norm = qm_norm.normalize(**norm_kwargs)

            if apply_scaling and scale_method:
                qm_norm = qm_norm.scale(method=scale_method)

        set_qm("qm_normalized", qm_norm)
        st.success("✅ Normalization complete.")

    except Exception as e:
        st.error(f"❌ Normalization failed: {e}")
        st.exception(e)

# ── Results preview ────────────────────────────────────────────────────────
qm_norm = st.session_state.get("qm_normalized")

if qm_norm is not None:
    st.divider()
    st.subheader("📊 Results")

    tab1, tab2 = st.tabs(["Before", "After"])
    with tab1:
        st.plotly_chart(
            intensity_boxplot(qm_input, "Before Normalization"),
            use_container_width=True,
        )
    with tab2:
        st.plotly_chart(
            intensity_boxplot(qm_norm, "After Normalization"),
            use_container_width=True,
        )

    st.info("👉 Proceed to **4. Batch Correction** in the sidebar.")
