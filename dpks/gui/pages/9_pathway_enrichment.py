"""
Page 9 — Pathway Enrichment
Gene-set enrichment analysis (over-representation test) on significant proteins.
"""

import sys, os
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import streamlit as st
import pandas as pd
import gseapy as gp

from utils.state import render_sidebar, require_step, get_qm, set_qm

st.set_page_config(page_title="9. Pathway Enrichment — DPKS GUI", layout="wide")
render_sidebar()

st.title("9. Pathway Enrichment")
st.markdown(
    "Perform gene-set over-representation analysis on proteins identified as significant "
    "by statistical testing or Importance-based feature importance. "
    "Uses the **gseapy** library under the hood via DPKS."
)

if not require_step("qm_explained", "Explainable ML", "8. Explainable ML"):
    st.stop()

qm_input = get_qm("qm_explained")

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

st.divider()

# ── Protein filter ─────────────────────────────────────────────────────────
st.subheader("🔬 Protein Selection for Enrichment")

filter_mode = st.radio(
    "Filter proteins by",
    options=["Adjusted p-value", "Importance value", "No filter (use all annotated proteins)"],
    horizontal=True,
)

ann = qm_input.row_annotations

explain_comparisons = st.session_state.get("explain_comparisons", [])
stat_comparisons = st.session_state.get("stat_comparisons", [])

filter_kwargs = {}

if filter_mode == "Importance value" and explain_comparisons:
    importance_comparison = st.selectbox(
        "Importance comparison",
        options=explain_comparisons,
        format_func=lambda c: f"Group {c[0]} vs Group {c[1]}",
    )
    g1, g2 = importance_comparison
    importance_col = f"MeanImportance{g1}-{g2}"
    if importance_col in ann.columns:
        default_cutoff = float(ann[importance_col].quantile(0.75)) if importance_col in ann.columns else 0.0
        importance_cutoff = st.slider(
            f"Min. mean |Importance| cutoff",
            min_value=0.0,
            max_value=float(ann[importance_col].max()) if importance_col in ann.columns else 1.0,
            value=max(0.0, default_cutoff),
            step=0.001,
            format="%.4f",
        )
        filter_kwargs = dict(filter_Importance=True, Importance_column=importance_col, Importance_cutoff=importance_cutoff)
    else:
        st.warning(f"Column `{importance_col}` not found. Check the Explainable ML step.")

elif filter_mode == "Adjusted p-value" and stat_comparisons:
    pval_comparison = st.selectbox(
        "P-value comparison",
        options=stat_comparisons,
        format_func=lambda c: f"Group {c[0]} vs Group {c[1]}",
    )
    g1, g2 = pval_comparison
    pval_col = f"CorrectedPValue{g1}-{g2}"
    if pval_col in ann.columns:
        pval_cutoff = st.slider(
            "Adjusted p-value cutoff",
            min_value=0.0001, max_value=0.2, value=0.05, step=0.001, format="%.3f",
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
