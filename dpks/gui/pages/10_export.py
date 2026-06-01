"""
Page 10 — Export
Download results and auto-generated pipeline code.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import streamlit as st
import pandas as pd
from dpks.gui.utils.io import df_to_tsv_bytes

st.set_page_config(page_title="10. Export — DPKS GUI", layout="wide")

st.title("10. Export")
st.markdown(
    "Download your results and a **reproducible Python script** "
    "that recreates the exact pipeline you built in this session."
)

st.divider()

# ── Data downloads ─────────────────────────────────────────────────────────
st.subheader("📥 Download Results")

steps = {
    "Filtered data":              ("qm_filtered",   "dpks_filtered.tsv"),
    "Normalised data":            ("qm_normalized",  "dpks_normalized.tsv"),
    "Batch-corrected data":       ("qm_corrected",   "dpks_corrected.tsv"),
    "Imputed data":               ("qm_imputed",     "dpks_imputed.tsv"),
    "Protein quantification":     ("qm_quantified",  "dpks_proteins.tsv"),
    "Statistical results":        ("qm_compared",    "dpks_statistics.tsv"),
    "SHAP / ML results":          ("qm_explained",   "dpks_shap.tsv"),
}

for label, (state_key, filename) in steps.items():
    qm = st.session_state.get(state_key)
    col1, col2 = st.columns([4, 2])
    col1.markdown(f"**{label}**")
    if qm is not None:
        try:
            tsv_bytes = df_to_tsv_bytes(qm.to_df())
            col2.download_button(
                label=f"⬇️ {filename}",
                data=tsv_bytes,
                file_name=filename,
                mime="text/tab-separated-values",
                key=f"dl_{state_key}",
            )
        except Exception as e:
            col2.error(f"Error: {e}")
    else:
        col2.caption("Not available")

# ── Enrichment results ─────────────────────────────────────────────────────
enr = st.session_state.get("enrich_result")
col1, col2 = st.columns([4, 2])
col1.markdown("**Pathway enrichment results**")
if enr is not None:
    try:
        enr_df = enr.results if hasattr(enr, "results") else pd.DataFrame()
        if not enr_df.empty:
            col2.download_button(
                label="⬇️ dpks_enrichment.tsv",
                data=df_to_tsv_bytes(enr_df),
                file_name="dpks_enrichment.tsv",
                mime="text/tab-separated-values",
                key="dl_enrich",
            )
        else:
            col2.caption("Empty results")
    except Exception as e:
        col2.error(f"Error: {e}")
else:
    col2.caption("Not available")

st.divider()

st.subheader("📋 Automated Methods")
st.text("This section will automatically generate a methods section of the results.")

st.divider()

# ── Session summary ────────────────────────────────────────────────────────
st.subheader("📋 Session Summary")

summary_rows = []
for key, label in {
    "qm_loaded":     "1. Data Loading",
    "qm_filtered":   "2. Filtering",
    "qm_normalized": "3. Normalization & Scaling",
    "qm_corrected":  "4. Batch Correction",
    "qm_quantified": "5. Quantification",
    "qm_imputed":    "6. Imputation",
    "qm_compared":   "7. Statistical Comparison",
    "qm_explained":  "8. Explainable ML",
    "enrich_result": "9. Pathway Enrichment",
}.items():
    done = st.session_state.get(key) is not None
    qm = st.session_state.get(key)
    detail = ""
    if done and hasattr(qm, "num_rows"):
        detail = f"{qm.num_rows} rows × {qm.num_samples} samples"
    elif done and key == "enrich_result":
        try:
            n = len(qm.results) if hasattr(qm, "results") else "?"
            detail = f"{n} enriched terms"
        except Exception:
            detail = "complete"
    summary_rows.append({
        "Step": label,
        "Status": "✅ Done" if done else "⬜ Not run",
        "Details": detail,
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

st.session_state["qm_exported"] = True
