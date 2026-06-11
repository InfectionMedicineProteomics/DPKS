"""
Page 9 — Pathway Enrichment
Gene-set enrichment analysis (over-representation test) on significant proteins.
"""

import copy

import gseapy as gp
import pandas as pd
import streamlit as st

from dpks.gui.utils.io import df_to_tsv_bytes
from dpks.gui.utils.state import require_step, get_qm, set_qm

st.set_page_config(page_title="9. Pathway Enrichment — DPKS GUI", layout="wide")

st.title("9. Pathway Enrichment")
st.markdown(
    "Perform gene-set over-representation analysis on proteins identified as significant "
    "by statistical testing or Importance-based feature importance. "
    "Uses the **gseapy** library under the hood via DPKS."
)

if not require_step("qm_explained", "Explainable ML", "8. Explainable ML"):
    st.stop()

qm_input = get_qm("qm_explained")

print(qm_input.row_annotations)

print(qm_input.annotated)
# Ensure proteins are annotated with gene names (needed for enrichment)
if not qm_input.annotated:
    with st.spinner("Annotating proteins with gene names via UniProt…"):
        try:
            qm_input = copy.deepcopy(qm_input)
            qm_input.annotate()
            set_qm("qm_explained", qm_input)
            st.success("✅ Protein annotation complete.")
        except Exception as e:
            st.warning(f"⚠️ Annotation failed (UniProt lookup may be unavailable): {e}")

st.divider()

# ── Enrichment parameters ──────────────────────────────────────────────────
st.subheader("⚙️ Enrichment Parameters")

col1, col2 = st.columns(2)

with col1:
    enrich_method = st.selectbox(
        "Enrichment method",
        options=["overreptest", "enrichr_overreptest"],
        help=(
            "**overreptest** — local over-representation test (no internet required). "
            "**enrichr_overreptest** — query the Enrichr API (requires internet)."
        ),
    )

    organism = st.selectbox(
        "Organism",
        options=["Human", "Mouse", "Yeast"],
        disabled=(enrich_method == "overreptest"),
        help="Only used for Enrichr queries.",
    )

with col2:
    available_libraries = gp.get_library_name(organism=organism)

    selected_libraries = st.multiselect(
        "Gene set libraries",
        options=available_libraries,
        default=["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"],
    )
    st.session_state['enrich_result'] = None

st.divider()

# ── Protein filter ─────────────────────────────────────────────────────────
st.subheader("🔬 Protein Selection for Enrichment")

filter_mode = st.radio(
    "Filter proteins by",
    options=["Adjusted p-value", "Importance value", "No filter (use all annotated proteins)"],
    horizontal=True,
)

row_annotations = qm_input.row_annotations

explain_comparison = st.session_state.get("explain_comparison", [])
stat_comparison = st.session_state.get("stat_comparison", [])

filter_kwargs = {}

if filter_mode == "Importance value" and explain_comparison:
    st.markdown("**Current comparison:**")
    st.markdown(f"- Group **{explain_comparison[0]}** vs Group **{explain_comparison[1]}**")
    g1, g2 = explain_comparison
    importance_col = f"MeanImportance{g1}-{g2}"
    if importance_col in row_annotations.columns:
        default_cutoff = float(
            row_annotations[importance_col].quantile(0.75)) if importance_col in row_annotations.columns else 0.0
        importance_cutoff = st.number_input(
            f"Min. mean |Importance| cutoff",
            min_value=0.0,
            max_value=float(
                row_annotations[importance_col].max()) if importance_col in row_annotations.columns else 1.0,
            value=max(0.0, default_cutoff),
            step=0.001,
            format="%.4f",
        )
        filter_kwargs = dict(filter_importance=True, importance_column=importance_col,
                             importance_cutoff=importance_cutoff)
    else:
        st.warning(f"Column `{importance_col}` not found. Check the Explainable ML step.")

elif filter_mode == "Adjusted p-value" and stat_comparison:
    st.markdown("**Current comparison:**")
    st.markdown(f"- Group **{stat_comparison[0]}** vs Group **{stat_comparison[1]}**")
    g1, g2 = stat_comparison
    pval_col = f"CorrectedPValue{g1}-{g2}"
    if pval_col in row_annotations.columns:
        pval_cutoff = st.number_input(
            "Adjusted p-value cutoff",
            min_value=0.0001, max_value=1.0, value=0.05, step=0.001, format="%.3f",
        )
        filter_kwargs = dict(filter_pvalue=True, pvalue_column=pval_col, pvalue_cutoff=pval_cutoff)
    else:
        st.warning(f"Column `{pval_col}` not found. Check the Statistical Comparison step.")

st.divider()

if st.button(
    "▶️ Run Pathway Enrichment",
    type="primary",
    disabled=len(selected_libraries) == 0,
):
    try:
        with st.spinner("Running enrichment analysis…"):
            enrich_kwargs = dict(
                method=enrich_method,
                libraries=selected_libraries,
                **filter_kwargs,
            )
            if enrich_method == "enrichr_overreptest":
                enrich_kwargs["organism"] = organism

            enr = qm_input.enrich(**enrich_kwargs)

        st.session_state["enrich_result"] = enr
        st.success("✅ Enrichment analysis complete.")

    except Exception as e:
        st.error(f"❌ Enrichment failed: {e}")
        st.exception(e)

# ── Results ────────────────────────────────────────────────────────────────
enr = st.session_state.get("enrich_result")

if enr is not None:
    st.divider()
    st.subheader("📊 Enrichment Results")

    tsv_bytes = df_to_tsv_bytes(enr.res2d)

    comparison = stat_comparison or explain_comparison

    databases = "_".join(selected_libraries)

    applied_filter_field = filter_mode.lower()

    if "p-value" in applied_filter_field:
        applied_filter_field = "pvalue"
    elif "importance" in applied_filter_field:
        applied_filter_field = "importance"

    filename = st.text_input(
        label="File name",
        value=f"dpks_enrich_{comparison[0]}_{comparison[1]}_{databases}_{applied_filter_field}.tsv"
    )

    st.download_button(
        label=f"⬇️ Download",
        data=tsv_bytes,
        file_name=filename,
        mime="text/tab-separated-values",
    )

    try:
        results_df = enr.results if hasattr(enr, "results") else pd.DataFrame()

        if results_df.empty:
            st.warning("No enriched terms found with the current filters.")
        else:
            # Sort by adjusted p-value if available
            sort_col = "Adjusted P-value" if "Adjusted P-value" in results_df.columns else results_df.columns[0]
            results_df = results_df.sort_values(sort_col)

            sig_col = "Adjusted P-value"
            if sig_col in results_df.columns:
                n_sig = (results_df[sig_col] < 0.05).sum()
                st.metric("Significant terms (adj. p < 0.05)", int(n_sig))

            import plotly.express as px

            # Top 20 bubble/bar chart
            plot_df = results_df.head(20).copy()
            if "Term" in plot_df.columns and sig_col in plot_df.columns:
                import numpy as np

                plot_df["-log10(adj.p)"] = -np.log10(plot_df[sig_col].clip(lower=1e-300))
                fig = px.bar(
                    plot_df.sort_values("-log10(adj.p)"),
                    x="-log10(adj.p)",
                    y="Term",
                    orientation="h",
                    color="-log10(adj.p)",
                    color_continuous_scale="Blues",
                    title="Top 20 Enriched Terms (-log₁₀ adj. p-value)",
                )
                fig.update_layout(coloraxis_showscale=False, height=600)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(results_df, use_container_width=True)

    except Exception as e:
        st.error(f"Could not parse enrichment results: {e}")
        st.exception(e)

    st.info("👉 Proceed to **10. Export** in the sidebar.")
