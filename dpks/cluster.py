import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from dpks.fdr import DecoyCounter


class FeatureClustering:
    q_value: float
    distance_cutoff: float
    decoy_min_distances: np.ndarray
    target_min_distances: np.ndarray
    decoy_distance_matrix: np.ndarray
    target_distance_matrix: np.ndarray
    distance_df: pd.DataFrame

    def __init__(self, q_value: float = 0.01):
        self.q_value = q_value
        self.decoy_min_distances = None
        self.target_min_distances = None
        self.decoy_distance_matrix = None
        self.target_distance_matrix = None
        self.distance_cutoff = 0.0
        self.distance_df = None

    def fit_predict(self, X, background) -> ndarray:
        self.target_distance_matrix = _get_distance_matrix(X)
        self.decoy_distance_matrix = _get_distance_matrix(background)

        self.target_min_distances = _get_min_distances(self.target_distance_matrix)
        self.decoy_min_distances = _get_min_distances(self.decoy_distance_matrix)

        self.distance_df = pd.DataFrame(
            {
                "label": ["Decoy" for _ in range(len(self.decoy_min_distances))]
                + ["Target" for _ in range(len(self.target_min_distances))],
                "distance": np.concatenate(
                    (self.decoy_min_distances, self.target_min_distances)
                ),
            }
        )

        self.distance_df["distance_score"] = 1 - self.distance_df["distance"]

        self.distance_df["label_"] = np.where(
            self.distance_df["label"] == "Decoy", 0, 1
        )

        decoy_counts = DecoyCounter()

        self.distance_df["q_value"] = decoy_counts.q_values(
            self.distance_df["distance_score"], self.distance_df["label_"]
        )

        self.distance_cutoff = (
            self.distance_df[self.distance_df["q_value"] <= self.q_value]
            .sort_values("q_value", ascending=False)
            .head(1)["distance"]
            .values[0]
        )

        return _cluster(self.target_distance_matrix, self.distance_cutoff)


def _get_distance_matrix(X) -> np.ndarray:
    corr = spearmanr(X).correlation
    # corr = np.nan_to_num(corr, nan=0)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    distance_matrix = 1 - np.abs(corr)

    return distance_matrix


def _get_min_distances(X) -> np.ndarray:
    distances = []

    for i in range(X.shape[0]):
        first = X[i, :i]
        second = X[i, i + 1 :]

        combined = np.concatenate((first, second))

        distances.append(np.min(combined))

    return np.array(distances)


def _cluster(X, distance_cutoff) -> np.ndarray:
    dist_linkage = hierarchy.linkage(squareform(X), method="ward")

    return np.array(
        hierarchy.fcluster(dist_linkage, distance_cutoff, criterion="distance")
    )
