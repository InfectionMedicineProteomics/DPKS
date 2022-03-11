from abc import ABC, abstractmethod

import numpy as np  # type: ignore

from enum import Enum

from sklearn.base import TransformerMixin


class NormalizationMethod(Enum):
    MEAN = 1
    MEDIAN = 2


class Normalization(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TicNormalization(TransformerMixin):

    def __init__(self):

        pass

    def fit_transform(self, X, y=None, **fit_params):

        sample_sums = np.nansum(X, axis=0)

        median_signal = np.nanmedian(sample_sums)

        normalized_signal = (X / sample_sums[None, :]) * median_signal

        normalized_signal = np.log2(normalized_signal)

        return normalized_signal


class MedianNormalization(TransformerMixin):

    def __init__(self):

        pass

    def fit_transform(self, X, y=None, **fit_params):

        sample_medians = np.nanmedian(X, axis=0)

        mean_sample_median = np.mean(sample_medians)

        normalized_signal = (X / sample_medians[None, :]) * mean_sample_median

        normalized_signal = np.log2(normalized_signal)

        return normalized_signal
