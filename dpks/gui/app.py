from pathlib import Path

import streamlit as st
from PIL import Image

from dpks.gui.utils.state import render_sidebar

ASSETS = Path(__file__).parent / "assets"

icon = Image.open(ASSETS / "logo.png")

st.set_page_config(
    page_title="DPKS GUI",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    {"Home": [st.Page("pages/0_home.py", title="Home", icon="🏠")],
     "Data Processing": [
         st.Page("pages/1_data_loading.py", title="Data Loading", icon="📂"),
         st.Page("pages/2_filtering.py", title="Filtering", icon="🔬"),
         st.Page("pages/3_normalization.py", title="Normalization", icon="📐"),
         st.Page("pages/4_batch_correction.py", title="Batch Correction", icon="🗂️"),
         st.Page("pages/5_quantification.py", title="Quantification", icon="🩹"),
         st.Page("pages/6_imputation.py", title="Imputation", icon="⚖️"),
     ],
     "Analysis": [
         st.Page("pages/7_statistical_comparison.py", title="Statistical Comparison", icon="📊"),
         st.Page("pages/8_explainable_ml.py", title="Explainable ML", icon="🤖"),
         st.Page("pages/9_pathway_enrichment.py", title="Pathway Enrichment", icon="🧬"),
     ],
     "Output": [
         st.Page("pages/10_export.py", title="Export", icon="💾"),
     ],
     }
)

pg.run()
render_sidebar()
