from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np  # type: ignore

from enum import Enum


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


class MeanNormalization(Normalization):

    mean: np.ndarray
    stdev: np.ndarray
    log_transforms: np.ndarray

    def __init__(self, log_transform: bool = False, shape: Tuple[int, int] = (0, 0)):

        self.log_transform = log_transform
        self.mean = np.ndarray(dtype="f8", shape=(shape[0],))
        self.stdev = np.ndarray(dtype="f8", shape=(shape[0],))

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

    median: np.ndarray
    iqr: np.ndarray
    log_transforms: np.ndarray

    def __init__(self, log_transform: bool = False, shape: Tuple[int, int] = (0, 0)):

        self.log_transform = log_transform
        self.median = np.ndarray(dtype="f8", shape=(shape[0],))
        self.iqr = np.ndarray(dtype="f8", shape=(shape[0],))

    def fit(self, data: np.ndarray) -> None:

        if self.log_transform:

            data = np.log(data)

        self.median = np.nanmedian(data, axis=1).reshape((-1, 1))

        q75, q25 = np.nanpercentile(a=data, q=[75, 25])

        q75 = q75.reshape((-1, 1))
        q25 = q25.reshape((-1, 1))

        self.iqr = q75 - q25

    def transform(self, data: np.ndarray) -> np.ndarray:

        if self.log_transform:

            data = np.log(data)

        return (data - self.median) / self.iqr
