import seaborn as sns  # type: ignore # noqa: F401
from typing import TYPE_CHECKING, Any, Union, List, Tuple, Optional

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
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        shap_values: np.ndarray,
        X: np.ndarray,
        qm: QuantMatrix,
        cmap: Union[List, str],
        n_display: int = 5,
        jitter: float = 0.1,
        alpha: float = 0.75,
        feature_column: str = "Protein",
        order_by: str = "shap",
        n_bins=100,
    ):
        """Creates a SHAP summary plot-like figure.

        Args:
            shap_values (np.ndarray): shap values
            X (np.ndarray): feature values
            qm (QuantMatrix): quantmatrix

        Returns:
            plt.Figure: figure object
        """
        self.shap_values = shap_values
        self.order_by = order_by
        self.X = X
        self.qm = qm
        self.n_display = n_display
        self.cmap = cmap
        self.jitter = jitter
        self.alpha = alpha
        self.feature_column = feature_column
        self.n_bins = n_bins

        if not fig:
            self.fig, self.ax = plt.subplots(
                1, 1, figsize=(n_display * 1.5, n_display * 1)
            )
        else:
            self.fig = fig
            self.ax = ax

    def plot(self) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        plot_frame = pd.DataFrame(columns=["feature", "x", "y"])
        col_sum = np.mean(np.abs(self.shap_values), axis=0)

        if self.order_by == "shap":
            sort_index = np.argsort(-col_sum)
        elif self.order_by == "rank":
            sort_index = np.argsort(self.qm.row_annotations['FeatureRank'])

        feature_names = []
        for idx, feature_idx in enumerate(sort_index[0 : self.n_display]):
            feature_name = self.qm.quantitative_data.obs[self.feature_column][
                feature_idx
            ]
            feature_names.append(feature_name)
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
                            "feature_idx": idx,
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

        plot_frame = self._jitter(plot_frame)
        plot_frame["y"] = plot_frame["feature_idx"] + plot_frame["jitter"]
        c = plot_frame["fv"].map(colors)
        sns.violinplot(
            data=plot_frame,
            x="sv",
            y="feature",
            alpha=0.1,
            color="lightgray",
            linewidth=0,
            ax=self.ax,
            scale="count",
        )
        sns.scatterplot(
            x=plot_frame["sv"],
            y=plot_frame["y"],
            c=c,
            alpha=self.alpha,
            linewidth=0,
            ax=self.ax,
        )
        cb = self.fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=self.ax)

        cb.outline.set_visible(False)
        sns.despine(ax=self.ax, left=True, right=True, top=True)

        self.ax.set_xlabel("SHAP value")
        self.ax.set_ylabel("Feature")
        self.ax.set_yticks(range(self.n_display), feature_names)
        return self.fig, self.ax

    def _jitter(self, plot_frame):
        plot_frame["bin"] = pd.cut(
            plot_frame["sv"], bins=self.n_bins, labels=range(self.n_bins)
        )
        bins = plot_frame["bin"].value_counts()

        bins_desc = bins.index.values.tolist()
        bins_desc.reverse()
        jitters = []
        for row in plot_frame.iterrows():
            b = row[1].bin
            i = bins_desc.index(b)
            jitter = np.random.normal(scale=self.jitter) * i / self.n_bins
            jitters.append(jitter)
        plot_frame["jitter"] = jitters
        return plot_frame


class RFEPCA(Plot):
    def __init__(
        self,
        fig: matplotlib.figure.Figure,
        axs: List[matplotlib.axes.Axes],
        qm: QuantMatrix,
        cutoffs: List,
        cmap: Union[list, str],
    ):
        self.qm = qm
        self.cutoffs = cutoffs
        self.fig = fig
        if isinstance(axs, list):
            axs = np.array(axs)
        self.axs = axs
        self.cmap = cmap
        if not fig:
            self.fig, self.axs = plt.subplots(1, 1, figsize=(5, 5))

    def plot(self) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        qdf = self.qm.to_df()

        samples1 = self.qm.get_samples(group=1)
        samples2 = self.qm.get_samples(group=2)

        y = [1 for _ in samples1] + [2 for _ in samples2]
        if isinstance(self.cmap, list):
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "customcmap", self.cmap
            )
        else:
            cmap = plt.get_cmap(self.cmap)
        for ax, cutoff in zip(self.axs.ravel(), self.cutoffs):
            qdf_f = qdf[qdf.FeatureRank <= cutoff]
            qdf_f = qdf_f[samples1 + samples2]
            X = qdf_f.values.T
            X = sklearn.preprocessing.StandardScaler().fit_transform(X)
            pca = sklearn.decomposition.PCA(n_components=2)
            pca.fit(X)
            X = pca.transform(X)
            explained_variance = pca.explained_variance_ratio_

            sns.kdeplot(
                x=X[:, 0],
                y=X[:, 1],
                levels=5,
                hue=y,
                palette=cmap,
                ax=ax,
                fill=True,
                alpha=0.2,
            )

            sns.kdeplot(
                x=X[:, 0],
                y=X[:, 1],
                levels=5,
                hue=y,
                palette=cmap,
                linewidths=2,
                ax=ax,
            )

            sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=cmap, ax=ax)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"PC1 ({100*explained_variance[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({100*explained_variance[1]:.1f}%)")

        return self.fig, self.axs
