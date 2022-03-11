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


class MeanNormalization(Normalization):
    def __init__(self, log_transform: bool = False, shape: tuple = None):

        self.log_transform = log_transform
        self.mean = np.ndarray(dtype="f8", shape=(shape[0],))  # type: ignore
        self.stdev = np.ndarray(dtype="f8", shape=(shape[0],))  # type: ignore

    def fit(self, data: np.ndarray) -> None:

        if self.log_transform:
            data = np.log(data)

        self.mean = np.nanmean(data, axis=1).reshape((-1, 1))
        self.stdev = np.nanstd(data, axis=1).reshape((-1, 1))

    def transform(self, data: np.ndarray) -> np.ndarray:

        if self.log_transform:
            data = np.log(data)

        return (data - self.mean) / self.stdev


class MedianNormalization(Normalization):
    def __init__(self, log_transform: bool = False, shape: tuple = None):

        self.log_transform = log_transform
        self.median = np.ndarray(dtype="f8", shape=(shape[0],))  # type: ignore
        self.stdev = np.ndarray(dtype="f8", shape=(shape[0],))  # type: ignore

    def fit(self, data: np.ndarray) -> None:

        if self.log_transform:
            data = np.log(data)

        self.median = np.nanmedian(data, axis=1).reshape((-1, 1))
        self.stdev = np.nanstd(data, axis=1).reshape((-1, 1))

    def transform(self, data: np.ndarray) -> np.ndarray:

        if self.log_transform:
            data = np.log(data)

        return (data - self.median) / self.stdev
