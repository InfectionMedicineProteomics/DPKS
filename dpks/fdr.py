from __future__ import annotations

import numpy as np


from typing import Dict, Union, Tuple, Any

from joblib import dump, load

from typing import TYPE_CHECKING

import numba
import pandas as pd

class DecoyFeatures:
    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame) -> None:
        pass

    @property
    def features(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.decoy_features,
            columns=[f"decoy_{i}" for i in self.feature_names],
            index=self.data_.index,
        )


class ShuffleDecoyFeatures(DecoyFeatures):
    n_samples: int
    n_features: int
    decoy_features: np.ndarray
    data_: pd.DataFrame
    random_seed: int

    def __init__(
        self,
        n_samples: int,
        n_features: int,
        feature_names: list[str],
        random_seed: int = 0,
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.feature_names = feature_names
        self.decoy_features = np.zeros((n_samples, n_features))
        self.random_seed = random_seed

    def fit(self, X: pd.DataFrame) -> None:
        np.random.seed(self.random_seed)
        self.data_ = X

        for i in range(self.n_features):
            feature_slice = X.iloc[:, i].copy().values

            np.random.shuffle(feature_slice)

            self.decoy_features[:, i] = feature_slice


class MeanDecoyFeatures(DecoyFeatures):
    n_samples: int
    n_features: int
    decoy_features: np.ndarray
    data_: pd.DataFrame
    random_seed: int

    def __init__(
        self,
        n_samples: int,
        n_features: int,
        feature_names: list[str],
        random_seed: int = 0,
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.feature_names = feature_names
        self.decoy_features = np.zeros((n_samples, n_features))
        self.random_seed = random_seed

    def fit(self, X: pd.DataFrame) -> None:
        np.random.seed(self.random_seed)
        self.data_ = X

        for i in range(self.n_features):
            feature_slice = X.iloc[:, i]

            feature_mean = np.nanmean(feature_slice)
            feature_std = np.nanstd(feature_slice)

            decoy_feature = np.random.normal(
                loc=feature_mean,
                scale=feature_std,
                size=self.n_samples,
            )

            self.decoy_features[:, i] = decoy_feature


@numba.jit(nopython=True)
def _fast_distribution_q_value(target_values, decoy_values, pit):  # type: ignore
    target_area = np.trapz(target_values)

    decoy_area = np.trapz(decoy_values)

    if decoy_area == 0.0 and target_area == 0.0:
        q_value = 0.0

    else:
        decoy_area = decoy_area * pit

        q_value = decoy_area / (decoy_area + target_area)

    return q_value


def _fast_distribution_q_values(scores, target_function, decoy_function, pit):  # type: ignore
    q_values = np.ones((len(scores),), dtype=np.float64)

    max_score = np.max(scores)

    for i in range(len(scores)):
        integral_bounds = np.arange(scores[i], max_score, 0.1)

        target_data = np.interp(integral_bounds, target_function[0], target_function[1])

        decoy_data = np.interp(integral_bounds, decoy_function[0], decoy_function[1])

        q_value = _fast_distribution_q_value(target_data, decoy_data, pit)

        q_values[i] = q_value

    return q_values


class ScoreDistribution:
    pit: float
    X: np.ndarray
    y: np.ndarray
    target_spline: Tuple[np.ndarray, np.ndarray]
    decoy_spline: Tuple[np.ndarray, np.ndarray]
    target_scores: np.ndarray
    decoy_scores: np.ndarray

    def __init__(self, pit: float = 1.0, num_threads: int = 1):
        numba.set_num_threads(num_threads)

        self.pit = pit

    def fit(self, X: np.ndarray, y: np.ndarray) -> ScoreDistribution:
        self.X = X
        self.y = y

        self._estimate_bin_number()

        target_indices = np.argwhere(y == 1)
        decoy_indices = np.argwhere(y == 0)

        self.target_scores = X[target_indices]
        self.decoy_scores = X[decoy_indices]

        self.target_spline = self._fit_function(self.target_scores)
        self.decoy_spline = self._fit_function(self.decoy_scores)

        return self

    def min(self) -> float:
        return float(np.min(self.X))

    def max(self) -> float:
        return float(np.max(self.X))

    def _estimate_bin_number(self) -> None:
        hist, bins = np.histogram(self.X, bins="auto")

        self.num_bins = (bins[1:] + bins[:-1]) / 2

    def _fit_function(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hist, bins = np.histogram(scores, bins=self.num_bins)

        bin_centers = (bins[1:] + bins[:-1]) / 2

        return bin_centers, hist

    def q_values(self, X: np.ndarray) -> np.ndarray:
        return _fast_distribution_q_values(
            X, self.target_spline, self.decoy_spline, self.pit
        )


class DecoyCounter:
    pit: float

    def __init__(self, num_threads: int = 1, pit: float = 1.0) -> None:
        numba.set_num_threads(num_threads)

        self.pit = pit

    def q_values(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        sorted_score_indices = np.argsort(scores)[::-1]

        num_scores = sorted_score_indices.shape[0]

        q_values = np.zeros((num_scores,), dtype=np.float64)

        num_targets = 0
        num_decoys = 0

        for idx in sorted_score_indices:
            label = labels[idx]

            if label == 1:
                num_targets += 1

            else:
                num_decoys += 1

            decoy_count = num_decoys * self.pit

            q_value = decoy_count / (decoy_count + num_targets)

            q_values[idx] = q_value

        return q_values
