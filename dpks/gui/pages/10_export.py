"""
Page 10 — Export
Download results and auto-generated pipeline code.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import streamlit as st
import pandas as pd
from utils.state import render_sidebar, get_qm

st.set_page_config(page_title="10. Export — DPKS GUI", layout="wide")
render_sidebar()

st.title("10. Export")
st.markdown(
    "Download your results and a **reproducible Python script** "
    "that recreates the exact pipeline you built in this session."
)

st.divider()

# ── Helper to convert a DataFrame to a TSV bytes buffer ───────────────────
def df_to_tsv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, sep="\t", index=False)
    return buf.getvalue().encode("utf-8")


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

# ── Pipeline code generator ────────────────────────────────────────────────
st.subheader("🐍 Auto-generated Pipeline Script")
st.markdown(
    "The script below reproduces your pipeline. "
    "Copy it into a `.py` file and run it independently — no GUI needed."
)


def build_pipeline_code() -> str:
    """Reconstruct a DPKS Python script from session state choices."""
    lines = [
        '"""',
        "Auto-generated DPKS pipeline script.",
        "Edit the file paths below before running.",
        '"""',
        "",
        "import xgboost",
        "from dpks.quant_matrix import QuantMatrix",
        "",
        '# ── File paths ──────────────────────────────────────────────────────────',
        'quant_file   = "quant_data.tsv"   # <- update this path',
        'design_file  = "design_matrix.tsv"  # <- update this path',
        "",
        "qm = QuantMatrix(",
        "    quantification_file=quant_file,",
        "    design_matrix_file=design_file,",
        ")",
    ]

    # Filter
    if st.session_state.get("qm_filtered") is not None:
        lines += [
            "",
            "# ── Filtering ───────────────────────────────────────────────────────────",
            "qm = qm.filter(",
            "    peptide_q_value=0.01,",
            "    protein_q_value=0.01,",
            "    remove_decoys=True,",
            "    remove_contaminants=True,",
            "    remove_non_proteotypic=True,",
            ")",
        ]

    # Normalize
    if st.session_state.get("qm_normalized") is not None:
        lines += [
            "",
            "# ── Normalization ───────────────────────────────────────────────────────",
            'qm = qm.normalize(method="mean", log_transform=True)',
        ]

    # Batch correction
    if st.session_state.get("qm_corrected") is not None:
        lines += [
            "",
            "# ── Batch Correction ────────────────────────────────────────────────────",
            'qm = qm.correct(method="combat")',
        ]

    # Imputation
    if st.session_state.get("qm_imputed") is not None:
        lines += [
            "",
            "# ── Imputation ──────────────────────────────────────────────────────────",
            'qm = qm.impute(method="uniform_percentile", percentile=0.1)',
        ]

    # Quantification
    if st.session_state.get("qm_quantified") is not None:
        lines += [
            "",
            "# ── Quantification ──────────────────────────────────────────────────────",
            'qm = qm.quantify(method="maxlfq", threads=4)',
        ]

    # Statistical comparison
    stat_comparisons = st.session_state.get("stat_comparisons", [])
    if st.session_state.get("qm_compared") is not None and stat_comparisons:
        cmp_str = repr(stat_comparisons)
        lines += [
            "",
            "# ── Statistical Comparison ──────────────────────────────────────────────",
            "qm = qm.compare(",
            '    method="linregress",',
            f"    comparisons={cmp_str},",
            "    min_samples_per_group=2,",
            '    multiple_testing_correction_method="fdr_tsbh",',
            ")",
        ]

    # Explainable ML
    explain_comparisons = st.session_state.get("explain_comparisons", [])
    if st.session_state.get("qm_explained") is not None and explain_comparisons:
        cmp_str = repr(explain_comparisons)
        lines += [
            "",
            "# ── Explainable ML ──────────────────────────────────────────────────────",
            "clf = xgboost.XGBClassifier(",
            "    max_depth=2,",
            "    reg_lambda=2,",
            '    objective="binary:logistic",',
            "    seed=42,",
            ")",
            "qm = qm.explain(",
            "    clf,",
            f"    comparisons={cmp_str},",
            "    n_iterations=100,",
            "    downsample_background=True,",
            ")",
        ]

    # Enrichment
    if st.session_state.get("enrich_result") is not None:
        lines += [
            "",
            "# ── Pathway Enrichment ──────────────────────────────────────────────────",
            "qm = qm.annotate()",
            "enr = qm.enrich(",
            '    method="overreptest",',
            '    libraries=["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"],',
            "    filter_shap=True,",
            ")",
        ]

    # Export
    lines += [
        "",
        "# ── Export ──────────────────────────────────────────────────────────────",
        'qm.write("dpks_results.tsv")',
        "",
        "print('Pipeline complete!')",
    ]

    return "\n".join(lines)


pipeline_code = build_pipeline_code()
st.code(pipeline_code, language="python")

st.download_button(
    label="⬇️ Download pipeline script (.py)",
    data=pipeline_code.encode("utf-8"),
    file_name="dpks_pipeline.py",
    mime="text/x-python",
)

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
