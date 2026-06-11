from pathlib import Path

import pandas as pd
import streamlit as st

from dpks.gui.utils.state import set_qm

st.set_page_config(page_title="1. Data Loading — DPKS GUI", layout="wide")

st.title("1. Data Loading")
st.markdown(
    """
    Upload your **quantification file** and **design matrix** to initialise the pipeline.
    Optionally provide a **FASTA file** to annotate proteins with descriptive labels.

    #### Expected file formats
    - **Quantification file** — tab-separated (`.tsv`) with precursor/peptide rows and
      sample columns matching the `sample` column of the design matrix.
    - **Design matrix** — tab-separated (`.tsv`) with at minimum the columns
      `sample`, `group`, and `batch` (batch is optional or can be set to a constant if no batch correction is needed).
    """
)

st.divider()

# ── File uploaders ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Quantification File")
    quant_file = st.file_uploader(
        "Upload quantification TSV",
        type=["tsv", "txt", "csv", "parquet"],
        help="Tab-separated file from GPS, DIA-NN, or similar tools.",
    )

with col2:
    st.subheader("Design Matrix")
    design_file = st.file_uploader(
        "Upload design matrix TSV",
        type=["tsv", "txt", "csv"],
        help="Tab-separated file mapping sample names to groups and batches.",
    )

st.divider()

# ── Optional settings ──────────────────────────────────────────────────────
with st.expander("⚙️ Advanced Options", expanded=False):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        quant_type = st.selectbox(
            "Quantification type",
            options=[
                "Standard",
                "DIA-NN"
            ],
            help="'Standard' for generic tab-separated output decribed in the DPKS documentation; 'DIA-NN' for DIA-NN .tsv or .parquet long report files.",
        )

    with col_b:
        diann_qvalue = st.number_input(
            "DIA-NN q-value threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Only applied when quant_type is 'DIA-NN'.",
            disabled=(quant_type != "DIA-NN"),
        )

    with col_c:
        fasta_file = st.file_uploader(
            "Annotation FASTA (optional)",
            type=["fasta", "fa", "txt"],
            help="Used to annotate proteins with descriptive labels.",
        )

st.divider()

# ── Load button ────────────────────────────────────────────────────────────
if st.button("🚀 Load Data", type="primary", disabled=(quant_file is None or design_file is None)):
    try:
        from dpks.quant_matrix import QuantMatrix

        sep = "\t"

        if quant_type == "DIA-NN":
            quant_type = "diann"
        elif quant_type == "Standard":
            quant_type = "standard"

        if quant_file:
            suffix = Path(quant_file.name).suffix

        if suffix == ".parquet":
            quant_df = pd.read_parquet(quant_file)
        else:
            quant_df = pd.read_csv(quant_file, sep=sep)

        design_df = pd.read_csv(design_file, sep=sep)

        if "sample" not in design_df:
            st.error(
                "❌ 'sample' column not found in design matrix. Please ensure the design matrix contains a 'sample' column."
            )

        init_kwargs = dict(
            quantification_file=quant_df,
            design_matrix_file=design_df,
            quant_type=quant_type,
        )

        if quant_type == "diann":
            init_kwargs["diann_qvalue"] = diann_qvalue

        # Save FASTA to a temp file if provided
        if fasta_file is not None:
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
                tmp.write(fasta_file.read())
                init_kwargs["annotation_fasta_file"] = tmp.name

        with st.spinner("Initialising QuantMatrix…"):
            qm = QuantMatrix(**init_kwargs)

        set_qm("qm_loaded", qm)
        st.success(
            f"✅ Data loaded successfully! "
            f"**{qm.num_rows}** precursor rows × **{qm.num_samples}** samples."
        )

    except ImportError:
        st.error(
            "❌ DPKS is not installed. Run `pip install dpks` and restart the app."
        )
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.exception(e)

# ── Preview ────────────────────────────────────────────────────────────────
qm = st.session_state.get("qm_loaded")

if qm is not None:
    st.divider()
    st.subheader("📊 Data Preview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precursor rows", qm.num_rows)
    m2.metric("Samples", qm.num_samples)
    m3.metric("Groups", qm.sample_annotations["group"].nunique())
    m4.metric("Proteins", len(qm.proteins))

    tab1, tab2 = st.tabs(["Quantification (first 100 rows)", "Design Matrix"])

    with tab1:
        st.dataframe(qm.to_df().head(100), width="stretch")

    with tab2:
        st.dataframe(qm.sample_annotations.reset_index(drop=True), width="stretch")

    st.info("👉 Proceed to **2. Filtering** in the sidebar.")
