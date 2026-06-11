"""
Page 7 — Statistical Comparison
Differential abundance analysis between sample groups.
"""

import copy

import streamlit as st

from dpks.gui.utils.io import df_to_tsv_bytes
from dpks.gui.utils.plots import volcano_plot
from dpks.gui.utils.state import require_step, get_qm, set_qm

st.set_page_config(page_title="7. Statistical Comparison — DPKS GUI", layout="wide")

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
        options=["linregress", "ttest", "ttest_paired", "anova", "fast_ols"],
        help=(
            "**linregress** — linear regression (recommended for most DIA data). "
            "**ttest** — Student's t-test. "
            "**ttest_paired** — paired t-test. "
            "**anova** — one-way ANOVA."
            "**fast_ols** — fast ordinary least squares regression."
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
st.subheader("➕ Set Comparison")
st.markdown(
    "A comparison is a pair `(group_A, group_B)` — fold changes are reported as A vs B. "
    "Groups are identified by their value in the design matrix."
)

if "comparison" not in st.session_state:
    st.session_state["comparison"] = None

cc1, cc2, cc3 = st.columns([2, 2, 1])
with cc1:
    new_g1 = st.selectbox("Group A (numerator)", options=groups, key="cmp_g1")
with cc2:
    remaining = [g for g in groups if g != new_g1]
    new_g2 = st.selectbox("Group B (denominator)", options=remaining, key="cmp_g2")
with cc3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Set comparison"):
        pair = (new_g1, new_g2)
        st.session_state["comparison"] = pair
        st.session_state["qm_compared"] = None

if st.session_state["comparison"]:
    st.markdown("**Current comparison:**")
    cmp = st.session_state["comparison"]
    st.markdown(f"- Group **{cmp[0]}** vs Group **{cmp[1]}**")
else:
    st.warning("No comparisons defined yet.")

st.divider()

# ── Run ────────────────────────────────────────────────────────────────────
comparison = st.session_state.get("comparison", ())

if st.button(
    "▶️ Run Statistical Comparison",
    type="primary",
    disabled=comparison is None,
):
    try:
        with st.spinner("Running differential abundance tests…"):
            qm_compared = copy.deepcopy(qm_input)
            qm_compared = qm_compared.compare(
                method=stat_method,
                comparison=comparison,
                min_samples_per_group=int(min_samples),
                level=level,
                multiple_testing_correction_method=correction_method,
            )
            # Store the comparisons used alongside the result
            st.session_state["stat_comparison"] = comparison

        set_qm("qm_compared", qm_compared)
        st.success(f"✅ Comparison complete.")

    except Exception as e:
        st.error(f"❌ Statistical comparison failed: {e}")
        st.exception(e)

# ── Results preview ────────────────────────────────────────────────────────
qm_compared = st.session_state.get("qm_compared")

if qm_compared is not None:
    st.divider()
    st.subheader("📊 Results")

    stored_comparison = st.session_state.get("stat_comparison", ())

    tsv_bytes = df_to_tsv_bytes(qm_compared.to_df())

    filename = st.text_input(
        label="File name",
        value=f"dpks_compared_{stored_comparison[0]}_{stored_comparison[1]}.tsv"
    )

    st.download_button(
        label=f"⬇️ Download",
        data=tsv_bytes,
        file_name=filename,
        mime="text/tab-separated-values",
    )

    col_fc, col_p = st.columns(2)
    fc_thresh = col_fc.slider(
        "Fold-change threshold (|log₂FC|)",
        min_value=0.0, max_value=5.0, value=1.0, step=0.1,
        key=f"fc_{stored_comparison}",
    )
    p_thresh = col_p.slider(
        "Adjusted p-value threshold",
        min_value=0.001, max_value=0.2, value=0.05, step=0.001,
        format="%.3f",
        key=f"pv_{stored_comparison}",
    )
    st.plotly_chart(
        volcano_plot(
            qm_compared, stored_comparison,
            fc_threshold=fc_thresh,
            pval_threshold=p_thresh,
        ),
        use_container_width=True,
    )

    with st.expander("View results table (first 200 rows)"):
        st.dataframe(qm_compared.row_annotations.head(200), use_container_width=True)

    st.info("👉 Proceed to **8. Explainable ML** in the sidebar.")
