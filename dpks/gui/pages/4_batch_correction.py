"""
Page 4 — Batch Correction
Correct for systematic batch effects using ComBat or mean correction.
"""

import sys, os

from utils.io import df_to_tsv_bytes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import streamlit as st
from dpks.gui.utils.state import render_sidebar, require_step, get_qm, set_qm
from dpks.gui.utils.plots import intensity_boxplot

st.set_page_config(page_title="4. Batch Correction — DPKS GUI", layout="wide")

st.title("4. Batch Correction")
st.markdown(
    "Correct for systematic batch effects in your data. "
    "This step requires a `batch` column in your design matrix."
)

if not require_step("qm_normalized", "Normalization & Scaling", "3. Normalization & Scaling"):
    st.stop()

qm_input = get_qm("qm_normalized")

# ── Check for batch column ─────────────────────────────────────────────────
has_batch_col = "batch" in qm_input.sample_annotations.columns
if not has_batch_col:
    st.warning(
        "⚠️ No `batch` column found in the design matrix. "
        "Batch correction will be skipped. "
        "You can still proceed — the normalised data will be passed through unchanged."
    )

if not has_batch_col:
    if st.button("Skip Batch Correction →", type="primary"):
        set_qm("qm_corrected", copy.deepcopy(qm_input))
        st.success("✅ Skipped — normalised data passed through to next step.")
        st.info("👉 Proceed to **5. Quantification** in the sidebar.")
    st.stop()

# ── Batch summary ──────────────────────────────────────────────────────────
batches = qm_input.get_batches()
unique_batches = list(set(batches))

st.divider()
st.subheader("🗂️ Batch Summary")

batch_counts = qm_input.sample_annotations["batch"].value_counts().reset_index()
batch_counts.columns = ["Batch", "Sample Count"]
st.dataframe(batch_counts, use_container_width=False)

st.divider()

# ── Batch correction parameters ──────────────────────────────────────────────────
st.subheader("⚙️ Batch Correction Parameters")

col1, col2 = st.columns(2)

with col1:
    correction_method = st.selectbox(
        "Batch correction method",
        options=["mean", "combat"],
        help=(
            "**mean** — subtract per-batch mean relative to a reference batch. "
            "**combat** — empirical Bayes batch correction (recommended for most cases)."
        ),
    )

with col2:
    reference_batch = None
    if correction_method == "mean":
        reference_batch = st.selectbox(
            "Reference batch",
            options=unique_batches,
            help="The batch whose mean will be used as the reference.",
        )

st.divider()

# ── Skip option ────────────────────────────────────────────────────────────
col_run, col_skip = st.columns([2, 1])

with col_run:
    run_correction = st.button("▶️ Apply Batch Correction", type="primary")

with col_skip:
    skip_correction = st.button("⏭️ Skip this step")

if skip_correction:
    set_qm("qm_corrected", copy.deepcopy(qm_input))
    st.success("✅ Skipped — normalised data passed through unchanged.")

if run_correction:
    try:
        with st.spinner("Applying batch correction…"):
            qm_corrected = copy.deepcopy(qm_input)
            correct_kwargs = dict(method=correction_method)
            if correction_method == "mean" and reference_batch is not None:
                correct_kwargs["reference_batch"] = reference_batch

            qm_corrected = qm_corrected.correct(**correct_kwargs)

        set_qm("qm_corrected", qm_corrected)
        st.success(f"✅ Batch correction complete using **{correction_method}** method.")

    except Exception as e:
        st.error(f"❌ Batch correction failed: {e}")
        st.exception(e)

# ── Results preview ────────────────────────────────────────────────────────
qm_corrected = st.session_state.get("qm_corrected")

if qm_corrected is not None:
    st.divider()
    st.subheader("📊 Results")

    tsv_bytes = df_to_tsv_bytes(qm_corrected.to_df())

    filename = st.text_input(
        label="File name",
        value="dpks_corrected.tsv"
    )

    st.download_button(
        label=f"⬇️ Download",
        data=tsv_bytes,
        file_name=filename,
        mime="text/tab-separated-values",
    )

    tab1, tab2 = st.tabs(["Before Correction", "After Correction"])
    with tab1:
        st.plotly_chart(
            intensity_boxplot(qm_input, "Before Batch Correction"),
            use_container_width=True,
        )
    with tab2:
        st.plotly_chart(
            intensity_boxplot(qm_corrected, "After Batch Correction"),
            use_container_width=True,
        )

    st.info("👉 Proceed to **5. Quantification** in the sidebar.")
