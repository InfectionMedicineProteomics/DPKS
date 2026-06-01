"""
Page 6 — Quantification
Roll up precursor/peptide intensities to protein-level quantities.
"""

#TODO: Add an Annotate()

import sys, os

from dpks.gui.utils.io import df_to_tsv_bytes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import streamlit as st
from dpks.gui.utils.state import require_step, get_qm, set_qm
from dpks.gui.utils.plots import intensity_boxplot

st.set_page_config(page_title="5. Quantification — DPKS GUI", layout="wide")

st.title("5. Quantification")
st.markdown(
    "Summarise precursor/peptide-level intensities into **protein-level quantities** "
    "using Top-N or MaxLFQ methods."
)

if not require_step("qm_corrected", "Batch Correction", "4. Batch Correction"):
    st.stop()

qm_input = get_qm("qm_corrected")

st.divider()
st.subheader("⚙️ Quantification Parameters")

col1, col2 = st.columns(2)

with col1:
    quant_method = st.selectbox(
        "Quantification method",
        options=["maxlfq", "top_n"],
        help=(
            "**maxlfq** — MaxLFQ (IQ) algorithm (recommended). Uses relative quantification "
            "across samples, robust to missing values. "
            "**top_n** — sum or mean of the top N most intense precursors per protein."
        ),
    )

    level = st.selectbox(
        "Quantification level",
        options=["protein", "peptide"],
        help="Whether to roll up to protein or peptide level.",
    )

with col2:
    if quant_method == "maxlfq":
        threads = st.slider(
            "Threads",
            min_value=1, max_value=16, value=4, step=1,
            help="Number of parallel threads for MaxLFQ computation.",
        )
        minimum_subgroups = st.number_input(
            "Minimum subgroups",
            min_value=1, max_value=10, value=1,
            help="Minimum number of samples required in each group for quantification.",
        )
        top_n_maxlfq = st.number_input(
            "Top N precursors (0 = all)",
            min_value=0, max_value=50, value=0,
            help="Restrict MaxLFQ to the top N precursors per protein. 0 uses all.",
        )
    else:
        top_n = st.number_input(
            "Top N precursors",
            min_value=1, max_value=50, value=3,
            help="Number of top-intensity precursors to use per protein.",
        )
        summarization_method = st.selectbox(
            "Summarization method",
            options=["sum", "mean", "median"],
        )

st.divider()

if st.button("▶️ Run Quantification", type="primary"):
    try:
        with st.spinner("Quantifying proteins… (this may take a moment for MaxLFQ)"):
            qm_work = copy.deepcopy(qm_input)

            if quant_method == "maxlfq":
                qm_quantified = qm_work.quantify(
                    method="maxlfq",
                    level=level,
                    threads=int(threads),
                    minimum_subgroups=int(minimum_subgroups),
                    top_n=int(top_n_maxlfq),
                )
            else:
                qm_quantified = qm_work.quantify(
                    method="top_n",
                    level=level,
                    top_n=int(top_n),
                    summarization_method=summarization_method,
                )

        set_qm("qm_quantified", qm_quantified)
        st.success(
            f"✅ Quantification complete. "
            f"**{qm_quantified.num_rows}** proteins × **{qm_quantified.num_samples}** samples."
        )

    except Exception as e:
        st.error(f"❌ Quantification failed: {e}")
        st.exception(e)

# ── Results preview ────────────────────────────────────────────────────────
qm_quantified = st.session_state.get("qm_quantified")

if qm_quantified is not None:
    st.divider()
    st.subheader("📊 Results")

    tsv_bytes = df_to_tsv_bytes(qm_quantified.to_df())

    filename = st.text_input(
        label="File name",
        value="dpks_quantified.tsv"
    )

    st.download_button(
        label=f"⬇️ Download",
        data=tsv_bytes,
        file_name=filename,
        mime="text/tab-separated-values",
    )

    m1, m2 = st.columns(2)
    m1.metric("Proteins", qm_quantified.num_rows)
    m2.metric("Samples", qm_quantified.num_samples)

    st.plotly_chart(
        intensity_boxplot(qm_quantified, "Protein-level Intensity Distribution"),
        use_container_width=True,
    )

    with st.expander("View protein matrix (first 100 rows)"):
        st.dataframe(qm_quantified.to_df().head(100), use_container_width=True)

    st.info("👉 Proceed to **7. Imputation** in the sidebar.")
