"""
Page 2 — Filtering
Remove decoys, contaminants, non-proteotypic entries, and zero/sparse rows.
"""

import copy

import streamlit as st

from dpks.gui.utils.io import df_to_tsv_bytes
from dpks.gui.utils.plots import intensity_boxplot
from dpks.gui.utils.state import require_step, get_qm, set_qm

st.set_page_config(page_title="2. Filtering — DPKS GUI", layout="wide")

st.title("2. Filtering")
st.markdown(
    "Remove low-quality entries from the precursor matrix before downstream analysis."
)

if not require_step("qm_loaded", "Data Loading", "1. Data Loading"):
    st.stop()

qm_input = get_qm("qm_loaded")

st.divider()

# ── Filter parameters ──────────────────────────────────────────────────────
st.subheader("⚙️ Filter Parameters")

col1, col2 = st.columns(2)

with col1:
    peptide_q_value = st.slider(
        "Peptide q-value threshold",
        min_value=0.0, max_value=0.1, value=0.01, step=0.001, format="%.3f",
        help="Precursors with PeptideQValue above this threshold are removed.",
    )
    protein_q_value = st.slider(
        "Protein q-value threshold",
        min_value=0.0, max_value=0.1, value=0.01, step=0.001, format="%.3f",
        help="Precursors with ProteinQValue above this threshold are removed.",
    )
    remove_decoys = st.checkbox("Remove decoys", value=True)
    remove_contaminants = st.checkbox("Remove contaminants", value=True)

with col2:
    remove_non_proteotypic = st.checkbox(
        "Remove non-proteotypic peptides",
        value=True,
        help="Removes rows where the Protein field contains ';' (shared peptides).",
    )
    remove_zero_rows = st.checkbox(
        "Remove all-zero rows",
        value=True,
        help="Remove rows where all sample intensities are zero or NaN.",
    )
    remove_n_zero_rows = st.checkbox(
        "Remove rows with too many zeros",
        value=False,
    )
    max_n_zeros = None
    if remove_n_zero_rows:
        max_n_zeros = st.number_input(
            "Maximum number of zeros allowed per row",
            min_value=0,
            max_value=qm_input.num_samples,
            value=max(0, qm_input.num_samples - 2),
            step=1,
        )

st.divider()

# ── Apply button ───────────────────────────────────────────────────────────
if st.button("▶️ Apply Filtering", type="primary"):
    try:
        with st.spinner("Filtering…"):
            qm_filtered = copy.deepcopy(qm_input)
            kwargs = dict(
                peptide_q_value=peptide_q_value,
                protein_q_value=protein_q_value,
                remove_decoys=remove_decoys,
                remove_contaminants=remove_contaminants,
                remove_non_proteotypic=remove_non_proteotypic,
                remove_zero_rows=remove_zero_rows,
                remove_n_zero_rows=remove_n_zero_rows,
            )
            if remove_n_zero_rows and max_n_zeros is not None:
                kwargs["max_n_zeros"] = int(max_n_zeros)

            qm_filtered = qm_filtered.filter(**kwargs)

        set_qm("qm_filtered", qm_filtered)

        removed = qm_input.num_rows - qm_filtered.num_rows
        st.success(
            f"✅ Filtering complete. **{qm_filtered.num_rows}** rows retained "
            f"(**{removed}** removed, "
            f"{removed / max(qm_input.num_rows, 1) * 100:.1f}% of input)."
        )

    except Exception as e:
        st.error(f"❌ Filtering failed: {e}")
        st.exception(e)

# ── Results preview ────────────────────────────────────────────────────────
qm_filtered = st.session_state.get("qm_filtered")

if qm_filtered is not None:
    st.divider()
    st.subheader("📊 Results")

    tsv_bytes = df_to_tsv_bytes(qm_filtered.to_df())

    filename = st.text_input(
        label="File name",
        value="dpks_filtered.tsv"
    )

    st.download_button(
        label=f"⬇️ Download",
        data=tsv_bytes,
        file_name=filename,
        mime="text/tab-separated-values",
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows before", qm_input.num_rows)
    m2.metric("Rows after", qm_filtered.num_rows, delta=-(qm_input.num_rows - qm_filtered.num_rows))
    m3.metric("Proteins retained", len(qm_filtered.proteins))

    st.info("👉 Proceed to **3. Normalization & Scaling** in the sidebar.")

    with st.expander("View filtered data (first 100 rows)"):
        st.dataframe(qm_filtered.to_df().head(100), width="stretch")

    if st.button("▶️ Generate Figures", type="primary"):
        try:
            st.plotly_chart(
                intensity_boxplot(qm_filtered, title="Post-filter Intensity Distribution"),
                width="stretch",
            )
        except Exception as e:
            st.error(f"❌ Figure generation failed: {e}")
            st.exception(e)
