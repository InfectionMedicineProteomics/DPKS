"""
utils/plots.py — reusable plotting helpers for DPKS GUI pages.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def intensity_boxplot(qm, title: str = "Sample Intensity Distribution") -> go.Figure:
    """
    Box plot of log2 intensities per sample, coloured by group.

    Parameters
    ----------
    qm : QuantMatrix
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = qm.quantitative_data.to_df()
    samples = list(qm.sample_annotations["sample"])
    groups = list(qm.sample_annotations["group"].astype(str))
    group_map = dict(zip(samples, groups))

    melted = df[samples].melt(var_name="sample", value_name="intensity")
    melted["group"] = melted["sample"].map(group_map)
    melted = melted.dropna(subset=["intensity"])

    fig = px.box(
        melted,
        x="sample",
        y="intensity",
        color="group",
        title=title,
        labels={"intensity": "log₂ Intensity", "sample": "Sample"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def volcano_plot(
    qm,
    comparison: tuple,
    fc_col: str = None,
    pval_col: str = None,
    padj_col: str = None,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
) -> go.Figure:
    """
    Volcano plot for a given comparison tuple, e.g. (2, 1).

    Automatically infers column names from the comparison tuple if not provided.
    """
    g1, g2 = comparison

    if fc_col is None:
        # DPKS names columns like Log2FoldChange2-1
        fc_col = f"Log2FoldChange{g1}-{g2}"
    if pval_col is None:
        pval_col = f"PValue{g1}-{g2}"
    if padj_col is None:
        padj_col = f"CorrectedPValue{g1}-{g2}"

    ann = qm.row_annotations.copy()

    # Fall back gracefully if columns don't exist
    missing = [c for c in [fc_col, padj_col] if c not in ann.columns]
    if missing:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Columns not found: {missing}.<br>Run the comparison step first.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    protein_col = "Gene" if "Gene" in ann.columns else "Protein"
    ann = ann[[fc_col, padj_col, protein_col]].dropna()
    ann["-log10(adj.p)"] = -np.log10(ann[padj_col].clip(lower=1e-300))

    def _colour(row):
        sig = row[padj_col] < pval_threshold
        up = row[fc_col] > fc_threshold
        down = row[fc_col] < -fc_threshold
        if sig and up:
            return "Up"
        if sig and down:
            return "Down"
        return "NS"

    ann["Regulation"] = ann.apply(_colour, axis=1)
    colour_map = {"Up": "#e74c3c", "Down": "#3498db", "NS": "#bdc3c7"}

    fig = px.scatter(
        ann,
        x=fc_col,
        y="-log10(adj.p)",
        color="Regulation",
        color_discrete_map=colour_map,
        hover_name=protein_col,
        hover_data={fc_col: ":.3f", padj_col: ":.2e"},
        title=f"Volcano Plot — Group {g1} vs Group {g2}",
        labels={fc_col: "log₂ Fold Change", "-log10(adj.p)": "-log₁₀ adj. p-value"},
    )
    fig.add_vline(x=fc_threshold, line_dash="dash", line_color="grey")
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="grey")
    fig.add_hline(y=-np.log10(pval_threshold), line_dash="dash", line_color="grey")
    return fig


def importance_bar_chart(qm, comparison: tuple, top_n: int = 20) -> go.Figure:
    """
    Horizontal bar chart of mean Importance values for the top_n proteins.
    """
    g1, g2 = comparison
    shap_col = f"MeanImportance{g1}-{g2}"
    rank_col = f"MeanRank{g1}-{g2}"
    protein_col = "Gene" if "Gene" in qm.row_annotations.columns else "Protein"

    ann = qm.row_annotations.copy()
    if shap_col not in ann.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Importance values not found. Run the Explainable ML step first.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    top = (
        ann[[protein_col, shap_col, rank_col]]
        .dropna()
        .drop_duplicates(subset=protein_col)
        .nlargest(top_n, shap_col)
        .sort_values(shap_col)
    )

    fig = px.bar(
        top,
        x=shap_col,
        y=protein_col,
        orientation="h",
        title=f"Top {top_n} Proteins by Mean Importance — Group {g1} vs Group {g2}",
        labels={shap_col: "Mean |Importance|", protein_col: "Protein / Gene"},
        color=shap_col,
        color_continuous_scale="Reds",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig
