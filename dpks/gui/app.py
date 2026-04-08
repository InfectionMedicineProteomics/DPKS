import streamlit as st

st.set_page_config(
    page_title="DPKS GUI",
    page_icon="docs/img/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar pipeline status ──────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.title("🧬 DPKS Pipeline")
    st.sidebar.markdown("---")

    steps = {
        "qm_loaded":      ("1. Data Loading",            "pages/1_data_loading.py"),
        "qm_filtered":    ("2. Filtering",               "pages/2_filtering.py"),
        "qm_normalized":  ("3. Normalization & Scaling", "pages/3_normalization.py"),
        "qm_corrected":   ("4. Batch Correction",        "pages/4_batch_correction.py"),
        "qm_quantified":  ("5. Quantification",          "pages/5_quantification.py"),
        "qm_imputed":     ("6. Imputation",              "pages/6_imputation.py"),
        "qm_compared":    ("7. Statistical Comparison",  "pages/7_statistical_comparison.py"),
        "qm_explained":   ("8. Explainable ML",          "pages/8_explainable_ml.py"),
        "enrich_result":  ("9. Pathway Enrichment",      "pages/9_pathway_enrichment.py"),
        "qm_exported":    ("10. Export",                 "pages/10_export.py"),
    }

    for key, (label, _) in steps.items():
        done = st.session_state.get(key) is not None
        icon = "✅" if done else "⬜"
        st.sidebar.markdown(f"{icon} {label}")

    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Reset Pipeline", width="stretch"):
        for key in steps:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


render_sidebar()

st.title("DPKS — Data Processing Kitchen Sink")

st.markdown(
    """
    Welcome to the **DPKS GUI**, a browser-based interface for proteomics
    data processing, statistical analysis, and explainable machine learning
    built on the [DPKS](https://github.com/InfectionMedicineProteomics/DPKS) Python package.

    ### How to use this app

    Navigate through the pipeline steps using the **sidebar** (or the pages listed below).
    Each step builds on the previous one — start with **Data Loading** and work your way through.

    | Step | Description |
    |------|-------------|
    | 1. Data Loading | Upload your quantification file and design matrix |
    | 2. Filtering | Remove decoys, contaminants, and low-quality entries |
    | 3. Normalization & Scaling | Normalise intensities and optionally scale features |
    | 4. Batch Correction | Correct for batch effects (ComBat or mean correction) |
    | 5. Quantification | Roll up precursors to protein-level quantities |
    | 6. Imputation | Impute missing values |
    | 7. Statistical Comparison | Differential abundance analysis between groups |
    | 8. Explainable ML | Train classifiers and compute SHAP feature importances |
    | 9. Pathway Enrichment | Gene-set enrichment analysis on significant proteins |
    | 10. Export | Download results and auto-generated pipeline code |

    ---
    > **Tip:** The sidebar shows a ✅ next to each completed step so you can
    > track your progress at a glance.
    """
)

st.info("👈 Start by navigating to **1. Data Loading** in the sidebar.")
