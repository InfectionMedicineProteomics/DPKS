"""
Page 7 — Statistical Comparison
Differential abundance analysis between sample groups.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import streamlit as st
from utils.state import render_sidebar, require_step, get_qm, set_qm
from utils.plots import volcano_plot

st.set_page_config(page_title="7. Statistical Comparison — DPKS GUI", layout="wide")
render_sidebar()

st.title("7. Statistical Comparison")
st.markdown(
    "Test for differential protein abundance between groups using "
    "classical statistical methods with multiple-testing correction."
)

if not require_step("qm_imputed", "Imputation", "6. Imputation"):
    st.stop()

qm_input = get_qm("qm_imputed")

# ── Group info ─────────────────────────────────────────────────────────────
groups = sorted(qm_input.sample_annotations["group"].unique().tolist())

st.divider()
st.subheader("🔬 Group Overview")
st.dataframe(
    qm_input.sample_annotations[["sample", "group"]].reset_index(drop=True),
    use_container_width=False,
)

st.divider()

# ── Statistical parameters ─────────────────────────────────────────────────
st.subheader("⚙️ Statistical Parameters")

col1, col2 = st.columns(2)

with col1:
    stat_method = st.selectbox(
        "Statistical test",
        options=["linregress", "ttest", "ttest_paired", "anova"],
        help=(
            "**linregress** — linear regression (recommended for most DIA data). "
            "**ttest** — Student's t-test. "
            "**ttest_paired** — paired t-test. "
            "**anova** — one-way ANOVA."
        ),
    )

    correction_method = st.selectbox(
        "Multiple testing correction",
        options=["fdr_tsbh", "fdr_bh", "bonferroni", "holm"],
        help="**fdr_tsbh** is the two-stage Benjamini–Hochberg FDR (recommended).",
    )

with col2:
    min_samples = st.number_input(
        "Min. samples per group",
        min_value=1, value=2,
        help="Proteins quantified in fewer samples than this per group are excluded.",
    )

    level = st.selectbox(
        "Analysis level",
        options=["protein", "peptide"],
    )

# ── Comparison builder ─────────────────────────────────────────────────────
st.divider()
st.subheader("➕ Define Comparisons")
st.markdown(
    "Each comparison is a pair `(group_A, group_B)` — fold changes are reported as A vs B. "
    "Groups are identified by their integer value in the design matrix."
)

if "comparisons" not in st.session_state:
    st.session_state["comparisons"] = []

cc1, cc2, cc3 = st.columns([2, 2, 1])
with cc1:
    new_g1 = st.selectbox("Group A (numerator)", options=groups, key="cmp_g1")
with cc2:
    remaining = [g for g in groups if g != new_g1]
    new_g2 = st.selectbox("Group B (denominator)", options=remaining, key="cmp_g2")
with cc3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Add comparison"):
        pair = (int(new_g1), int(new_g2))
        if pair not in st.session_state["comparisons"]:
            st.session_state["comparisons"].append(pair)

if st.session_state["comparisons"]:
    st.markdown("**Current comparisons:**")
    for i, cmp in enumerate(st.session_state["comparisons"]):
        col_label, col_remove = st.columns([5, 1])
        col_label.markdown(f"- Group **{cmp[0]}** vs Group **{cmp[1]}**")
        if col_remove.button("✕", key=f"rm_cmp_{i}"):
            st.session_state["comparisons"].pop(i)
            st.rerun()
else:
    st.warning("No comparisons defined yet. Add at least one above.")

st.divider()

# ── Run ────────────────────────────────────────────────────────────────────
comparisons = st.session_state.get("comparisons", [])

if st.button(
    "▶️ Run Statistical Comparison",
    type="primary",
    disabled=len(comparisons) == 0,
):
    try:
        with st.spinner("Running differential abundance tests…"):
            qm_compared = copy.deepcopy(qm_input)
            qm_compared = qm_compared.compare(
                method=stat_method,
                comparisons=comparisons,
                min_samples_per_group=int(min_samples),
                level=level,
                multiple_testing_correction_method=correction_method,
            )
            # Store the comparisons used alongside the result
            st.session_state["stat_comparisons"] = comparisons

        set_qm("qm_compared", qm_compared)
        st.success(f"✅ Comparison complete for {len(comparisons)} comparison(s).")

    except Exception as e:
        st.error(f"❌ Statistical comparison failed: {e}")
        st.exception(e)

# ── Results preview ────────────────────────────────────────────────────────
qm_compared = st.session_state.get("qm_compared")

if qm_compared is not None:
    st.divider()
    st.subheader("📊 Results")

    stored_comparisons = st.session_state.get("stat_comparisons", comparisons)

    if stored_comparisons:
        tabs = st.tabs([f"Group {g1} vs {g2}" for g1, g2 in stored_comparisons])
        for tab, cmp in zip(tabs, stored_comparisons):
            with tab:
                col_fc, col_p = st.columns(2)
                fc_thresh = col_fc.slider(
                    "Fold-change threshold (|log₂FC|)",
                    min_value=0.0, max_value=5.0, value=1.0, step=0.1,
                    key=f"fc_{cmp}",
                )
                p_thresh = col_p.slider(
                    "Adjusted p-value threshold",
                    min_value=0.001, max_value=0.2, value=0.05, step=0.001,
                    format="%.3f",
                    key=f"pv_{cmp}",
                )
                st.plotly_chart(
                    volcano_plot(
                        qm_compared, cmp,
                        fc_threshold=fc_thresh,
                        pval_threshold=p_thresh,
                    ),
                    use_container_width=True,
                )

    with st.expander("View results table (first 200 rows)"):
        st.dataframe(qm_compared.row_annotations.head(200), use_container_width=True)

    st.info("👉 Proceed to **8. Explainable ML** in the sidebar.")
