import seaborn as sns  # type: ignore # noqa: F401
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class Plot:
    """the base class"""

    def __init__(self) -> None:
        """init"""

        pass

    def plot(self) -> None:
        """create the plot"""
        pass


class SHAPPlot(Plot):
    def __init__(
        self, shap_values: np.ndarray, X: np.ndarray, quantified_data: QuantMatrix
    ) -> plt.Figure:
        """init"""
        assert shap_values.shape[0] == X.shape[0]
        self.shap_values = shap_values
        self.X = X
        self.quantified_data = quantified_data

    def plot(self, n_display: int = 5):
        plot_frame = pd.DataFrame(columns=["feature", "x", "y"])
        col_sum = np.mean(np.abs(self.shap_values), axis=0)
        sort_index = np.argsort(-col_sum)

        for feature_idx in sort_index[0:n_display]:
            feature_name = self.quantified_data.quantitative_data.obs["Protein"][
                feature_idx
            ]
            sv = self.shap_values[:, feature_idx]
            fv = self.X[:, feature_idx]
            plot_frame = pd.concat(
                [
                    plot_frame,
                    pd.DataFrame(
                        data={
                            "feature": len(fv) * [feature_name],
                            "sv": sv,
                            "fv": fv,
                        }
                    ),
                ]
            )

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "customcmap", ["#ff4800", "#ff4040", "#a836ff", "#405cff", "#05c9fa"]
        )
        v = min(np.abs(plot_frame["fv"].min()), plot_frame["fv"].max())
        norm = matplotlib.colors.Normalize(vmin=-v, vmax=v)

        colors = {}
        for cval in plot_frame["fv"]:
            colors.update({cval: cmap(norm(cval))})

        fig = plt.figure(figsize=(7, 5))
        plt.axvline([0], c="#f5f5f5", zorder=-1)
        sns.stripplot(
            data=plot_frame,
            x="sv",
            y="feature",
            dodge=False,
            jitter=1,
            alpha=0.95,
            size=5,
            palette=colors,
            hue="fv",
        )
        sns.violinplot(
            data=plot_frame,
            x="sv",
            y="feature",
            alpha=0.1,
            color="lightgray",
            linewidth=0,
        )

        sns.despine(left=True, right=True, top=True)
        plt.xlabel("SHAP value")
        plt.ylabel("Feature")

        plt.gca().legend_.remove()
        divider = make_axes_locatable(plt.gca())
        ax_cb = divider.new_horizontal(size="3%", pad=0.05)
        fig.add_axes(ax_cb)
        cb = matplotlib.colorbar.ColorbarBase(
            ax_cb, cmap=cmap, norm=norm, orientation="vertical"
        )

        cb.outline.set_visible(False)
        cb.set_label("Feature value")
        self.fig = fig
        return fig
