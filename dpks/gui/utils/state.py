"""
utils/state.py — shared session-state helpers and sidebar renderer.
Import this at the top of every page:

    from utils.state import render_sidebar, require_step, get_qm, set_qm
"""

from pathlib import Path
from PIL import Image
import streamlit as st

ASSETS = Path(__file__).parent.parent / "assets"

# ── Pipeline step keys ────────────────────────────────────────────────────────
STEP_KEYS = {
    "qm_loaded":     "1. Data Loading",
    "qm_filtered":   "2. Filtering",
    "qm_normalized": "3. Normalization & Scaling",
    "qm_corrected":  "4. Batch Correction",
    "qm_quantified": "5. Quantification",
    "qm_imputed":    "6. Imputation",
    "qm_compared":   "7. Statistical Comparison",
    "qm_explained":  "8. Explainable ML",
    "enrich_result": "9. Pathway Enrichment",
    "qm_exported":   "10. Export",
}


def render_sidebar():
    """Render the pipeline progress in the sidebar."""

    logo_path = ASSETS / "logo.png"
    if logo_path.exists():
        st.logo(
            str(logo_path),
            size="large",
            link="https://github.com/InfectionMedicineProteomics/DPKS",
        )

    st.sidebar.title("DPKS Pipeline Steps")
    st.sidebar.markdown("---")
    for key, label in STEP_KEYS.items():
        done = st.session_state.get(key) is not None
        icon = "✅" if done else "⬜"
        st.sidebar.markdown(f"{icon} {label}")
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Pipeline", width="stretch"):
        for key in STEP_KEYS:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


def require_step(state_key: str, friendly_name: str, page_hint: str = "") -> bool:
    """
    Check that a prior pipeline step has been completed.
    Returns True if satisfied, otherwise renders an error and returns False.
    """
    if st.session_state.get(state_key) is None:
        hint = f" Navigate to **{page_hint}** first." if page_hint else ""
        st.error(f"⚠️ Required step not completed: **{friendly_name}**.{hint}")
        return False
    return True


def get_qm(state_key: str = "qm_loaded"):
    """Retrieve a QuantMatrix from session state."""
    return st.session_state.get(state_key)


def set_qm(state_key: str, qm):
    """Store a QuantMatrix in session state."""
    st.session_state[state_key] = qm
