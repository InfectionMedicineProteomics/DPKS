import seaborn as sns  # type: ignore # noqa: F401
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import matplotlib
import sklearn

import matplotlib.pyplot as plt
from matplotlib import cm


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
        self,
        fig,
        ax,
        shap_values: np.ndarray,
        X: np.ndarray,
        qm: QuantMatrix,
        cmap: Union[list, str],
        n_display: int = 5,
    ) -> plt.Figure:
        """Creates a SHAP summary plot-like figure.

        Args:
            shap_values (np.ndarray): shap values
            X (np.ndarray): feature values
            qm (QuantMatrix): quantmatrix

        Returns:
            plt.Figure: figure object
        """
        assert shap_values.shape[0] == X.shape[0]
        self.shap_values = shap_values
        self.X = X
        self.qm = qm
        self.n_display = n_display
        self.cmap = cmap

        if not fig:
            self.fig, self.ax = plt.subplots(
                1, 1, figsize=(n_display * 1.5, n_display * 1)
            )
        else:
            self.fig = fig
            self.ax = ax

    def plot(self):
        plot_frame = pd.DataFrame(columns=["feature", "x", "y"])
        col_sum = np.mean(np.abs(self.shap_values), axis=0)
        sort_index = np.argsort(-col_sum)

        for feature_idx in sort_index[0 : self.n_display]:
            feature_name = self.qm.quantitative_data.obs["Protein"][feature_idx]
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

        if isinstance(self.cmap, list):
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "customcmap", self.cmap
            )
        else:
            cmap = plt.get_cmap(self.cmap)

        v = min(np.abs(plot_frame["fv"].min()), plot_frame["fv"].max())
        norm = matplotlib.colors.Normalize(vmin=-v, vmax=v)

        colors = {}
        for cval in plot_frame["fv"]:
            colors.update({cval: cmap(norm(cval))})

        self.ax.axvline([0], c="#f5f5f5", zorder=-1)
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
            ax=self.ax,
        )
        self.ax.get_legend().remove()
        cb = self.fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=self.ax)
        sns.violinplot(
            data=plot_frame,
            x="sv",
            y="feature",
            alpha=0.1,
            color="lightgray",
            linewidth=0,
            ax=self.ax,
        )
        cb.outline.set_visible(False)
        sns.despine(ax=self.ax, left=True, right=True, top=True)
        self.ax.set_xlabel("SHAP value")
        self.ax.set_ylabel("Feature")

        return self.fig, self.ax
