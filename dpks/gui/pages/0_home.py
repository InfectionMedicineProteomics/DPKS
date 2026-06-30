import streamlit as st

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
