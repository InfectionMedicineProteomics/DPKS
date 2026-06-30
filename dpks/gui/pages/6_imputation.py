"""
Page 5 — Imputation
Impute missing values before protein quantification.
"""

import copy

import streamlit as st

from dpks.gui.utils.io import df_to_tsv_bytes
from dpks.gui.utils.state import require_step, get_qm, set_qm

st.set_page_config(page_title="6. Imputation — DPKS GUI", layout="wide")

st.title("6. Imputation")
st.markdown(
    "Impute missing (NaN / zero) intensity values. "
    "This step is optional — some quantification methods handle missing values internally."
)

if not require_step("qm_quantified", "Quantification", "4. Quantification"):
    st.stop()

qm_input = get_qm("qm_quantified")

# ── Missingness summary ────────────────────────────────────────────────────
import numpy as np

X = qm_input.quantitative_data.X
n_total = X.size
n_missing = int(np.sum(np.isnan(X)) + np.sum(X == 0))
n_zero = int(np.sum(X == 0))
pct_missing = n_missing / n_total * 100

st.divider()
st.subheader("🔍 Missingness Summary")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total values", f"{n_total:,}")
m2.metric("Missing / zero values", f"{n_missing:,}")
m3.metric("% Missing", f"{pct_missing:.1f}%")
m4.metric("Zero values", f"{n_zero:,}")

st.divider()

# ── Imputation parameters ──────────────────────────────────────────────────
st.subheader("⚙️ Imputation Parameters")

col1, col2 = st.columns(2)

with col1:
    impute_method = st.selectbox(
        "Imputation method",
        options=["uniform_percentile", "uniform_range", "neighborhood"],
        help=(
            "**uniform_percentile** — replace missing values with a random draw from "
            "[0, percentile] of the observed distribution. "
            "**uniform_range** — replace with a uniform random draw from [minvalue, maxvalue]."
            "**neighborhood** — replace with a random draw from the observed intensities of the nearest neighbors."
        ),
    )

with col2:
    if impute_method == "uniform_percentile":
        percentile = st.slider(
            "Percentile",
            min_value=0.01, max_value=0.5, value=0.1, step=0.01,
            help="Values are drawn from [0, this percentile of observed intensities].",
        )
    elif impute_method == "uniform_range":
        col_min, col_max = st.columns(2)
        minvalue = col_min.number_input("Min value", value=0, step=1)
        maxvalue = col_max.number_input("Max value", value=1, step=1)
    elif impute_method == "neighborhood":
        n_neighbors = st.number_input("Number of nearest neighbors", value=5, step=1)

st.divider()

# ── Skip / Apply ───────────────────────────────────────────────────────────
col_run, col_skip = st.columns([2, 1])

with col_run:
    run_impute = st.button("▶️ Apply Imputation", type="primary")

with col_skip:
    skip_impute = st.button("⏭️ Skip this step")

if skip_impute:
    set_qm("qm_imputed", copy.deepcopy(qm_input))
    st.success("✅ Skipped — data passed through unchanged.")

if run_impute:
    try:
        with st.spinner("Imputing missing values…"):
            qm_imputed = copy.deepcopy(qm_input)

            impute_kwargs = dict(method=impute_method)
            if impute_method == "uniform_percentile":
                impute_kwargs["percentile"] = float(percentile)
            elif impute_method == "uniform_range":
                impute_kwargs["minvalue"] = int(minvalue)
                impute_kwargs["maxvalue"] = int(maxvalue)
            elif impute_method == "neighborhood":
                impute_kwargs["n_neighbors"] = n_neighbors

            qm_imputed = qm_imputed.impute(**impute_kwargs)

        set_qm("qm_imputed", qm_imputed)
        st.success("✅ Imputation complete.")

    except Exception as e:
        st.error(f"❌ Imputation failed: {e}")
        st.exception(e)

# ── Results preview ────────────────────────────────────────────────────────
qm_imputed = st.session_state.get("qm_imputed")

if qm_imputed is not None:
    st.divider()
    st.subheader("📊 Results")

    tsv_bytes = df_to_tsv_bytes(qm_imputed.to_df())

    filename = st.text_input(
        label="File name",
        value="dpks_imputed.tsv"
    )

    st.download_button(
        label=f"⬇️ Download",
        data=tsv_bytes,
        file_name=filename,
        mime="text/tab-separated-values",
    )

    X_after = qm_imputed.quantitative_data.X
    n_missing_after = int(np.sum(np.isnan(X_after)) + np.sum(X_after == 0))

    ma1, ma2 = st.columns(2)
    ma1.metric("Missing / zero before", f"{n_missing:,}")
    ma2.metric("Missing / zero after", f"{n_missing_after:,}", delta=-(n_missing - n_missing_after))

    st.info("👉 Proceed to **6. Quantification** in the sidebar.")
