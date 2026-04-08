"""
Page 8 — Explainable Machine Learning
Train a classifier and compute Importance-based protein feature importances.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import streamlit as st
from utils.state import render_sidebar, require_step, get_qm, set_qm
from utils.plots import importance_bar_chart

st.set_page_config(page_title="8. Explainable ML — DPKS GUI", layout="wide")
render_sidebar()

st.title("8. Explainable Machine Learning")
st.markdown(
    "Train a classifier on the protein matrix and use **Feature Importance** estimation "
    "to identify which proteins drive group differences. "
    "Results are stored back on the `QuantMatrix` for downstream enrichment analysis."
)

if not require_step("qm_compared", "Statistical Comparison", "7. Statistical Comparison"):
    st.stop()

qm_input = get_qm("qm_compared")
stored_comparisons = st.session_state.get("stat_comparisons", [])

st.divider()

# ── Classifier selection ───────────────────────────────────────────────────
st.subheader("🤖 Classifier")

clf_name = st.selectbox(
    "Classifier",
    options=["Logistic Regression", "XGBoost", "Random Forest", "Gradient Boosting", "SVM"],
    help="The classifier used to discriminate between groups. In most cases, Logistic Regression is sufficient.",
)

def build_classifier(name: str, params: dict):
    """Instantiate a sklearn-compatible classifier from name and params."""
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            max_depth=params.get("max_depth", 2),
            reg_lambda=params.get("reg_lambda", 2),
            objective="binary:logistic",
            seed=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", 1),
        )
    elif name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None) or None,
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", 1),
        )
    elif name == "Gradient Boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            random_state=params.get("random_state", 42),
        )
    elif name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            penalty=params.get("penalty", "l2"),
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", 1),
            solver=params.get("solver", "liblinear"),
        )
    elif name == "SVM":
        from sklearn.svm import SVC
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            probability=True,
            random_state=params.get("random_state", 42),
        )

# ── Classifier hyperparameters ─────────────────────────────────────────────
with st.expander("⚙️ Classifier Hyperparameters", expanded=True):
    params = {}
    col1, col2 = st.columns(2)

    if clf_name == "XGBoost":
        params["max_depth"] = col1.slider("max_depth", 1, 10, 2)
        params["reg_lambda"] = col2.number_input("reg_lambda", 0.0, 10.0, 2.0, step=0.5)

    elif clf_name == "Random Forest":
        params["n_estimators"] = col1.slider("n_estimators", 10, 500, 100, step=10)
        max_depth_val = col2.slider("max_depth (0 = unlimited)", 0, 20, 0)
        params["max_depth"] = max_depth_val if max_depth_val > 0 else 0

    elif clf_name == "Gradient Boosting":
        params["n_estimators"] = col1.slider("n_estimators", 10, 500, 100, step=10)
        params["max_depth"] = col2.slider("max_depth", 1, 10, 3)

    elif clf_name == "Logistic Regression":
        params["C"] = col1.number_input(
            help="Lower values for stronger regularization", label="Regularization C", min_value=0.001, max_value=100.0, value=1.0)
        params["max_iter"] = col2.number_input("max_iter", 100, 5000, 1000, step=100)
        params['penalty'] = col1.selectbox("Penalty", ["l1", "l2", "elasticnet"])
        params['solver'] = "liblinear"

    elif clf_name == "SVM":
        params["C"] = col1.number_input("Regularization C", 0.001, 100.0, 1.0)
        params["kernel"] = col2.selectbox("Kernel", ["rbf", "linear", "poly"])

    params["random_state"] = st.number_input("Random state (seed)", 0, 9999, 42)

st.divider()

# ── Importance / explain parameters ──────────────────────────────────────────────
st.subheader("🔍 Explanation Parameters")

col3, col4 = st.columns(2)

with col3:
    n_iterations = st.slider(
        "Bootstrap iterations",
        min_value=10, max_value=500, value=50, step=10,
        help="More iterations → more stable Importance estimates. Increases runtime.",
    )

with col4:
    downsample_background = st.checkbox(
        "Downsample background",
        value=True,
        help="Randomly downsample the background reference for Importance (recommended).",
    )

feature_column = st.selectbox(
    "Feature column",
    options=["Protein", "Gene"] if "Gene" in qm_input.row_annotations.columns else ["Protein"],
    help="The column used to identify features.",
)

# ── Comparison selector ────────────────────────────────────────────────────
st.divider()
st.subheader("🎯 Select Comparisons for Explanation")

if not stored_comparisons:
    st.warning("No comparisons found. Complete Step 7 first.")
    st.stop()

selected_comparisons = st.multiselect(
    "Comparisons to explain",
    options=stored_comparisons,
    default=stored_comparisons,
    format_func=lambda c: f"Group {c[0]} vs Group {c[1]}",
)

st.divider()

if st.button(
    "▶️ Run Explainable ML",
    type="primary",
    disabled=len(selected_comparisons) == 0,
):
    try:
        clf = build_classifier(clf_name, params)

        with st.spinner(
            f"Training {clf_name} and computing Importance values "
            f"({n_iterations} iterations per comparison)…"
        ):
            qm_explained = copy.deepcopy(qm_input)
            qm_explained = qm_explained.explain(
                clf=clf,
                comparisons=selected_comparisons,
                n_iterations=int(n_iterations),
                downsample_background=downsample_background,
                feature_column=feature_column,
            )

        set_qm("qm_explained", qm_explained)
        st.session_state["explain_comparisons"] = selected_comparisons
        st.success("✅ Explainable ML complete. Importance values added to QuantMatrix.")

    except ImportError as e:
        st.error(
            f"❌ Missing dependency: {e}. "
            "Install it with `pip install xgboost` or `pip install scikit-learn`."
        )
    except Exception as e:
        st.error(f"❌ Explainable ML failed: {e}")
        st.exception(e)

# ── Results ────────────────────────────────────────────────────────────────
qm_explained = st.session_state.get("qm_explained")

if qm_explained is not None:
    st.divider()
    st.subheader("📊 Feature Importance Results")

    explain_comparisons = st.session_state.get(
        "explain_comparisons", selected_comparisons
    )

    tabs = st.tabs([f"Group {g1} vs {g2}" for g1, g2 in explain_comparisons])
    for tab, cmp in zip(tabs, explain_comparisons):
        with tab:
            top_n_display = st.slider(
                "Top N proteins to display",
                min_value=5, max_value=50, value=20,
                key=f"top_n_{cmp}",
            )
            st.plotly_chart(
                importance_bar_chart(qm_explained, cmp, top_n=top_n_display),
                use_container_width=True,
            )

    with st.expander("View full annotations table (first 200 rows)"):
        st.dataframe(qm_explained.row_annotations.head(200), use_container_width=True)

    st.info("👉 Proceed to **9. Pathway Enrichment** in the sidebar.")
