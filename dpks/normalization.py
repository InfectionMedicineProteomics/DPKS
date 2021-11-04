from abc import (
    ABC,
    abstractmethod
)

import numpy as np

from enum import Enum

class NormalizationMethod(Enum):
    MEAN = 1
    MEDIAN = 2


class Normalization(ABC):

    @abstractmethod
    def fit(self, data : np.ndarray) -> None:

        raise NotImplementedError

    @abstractmethod
    def transform(self, data : np.ndarray) -> np.ndarray:

        raise NotImplementedError


class MeanNormalization(Normalization):

    def __init__(self, log_transform : bool = False, shape : tuple = None):

        self.log_transform = log_transform
        self.mean = np.ndarray(dtype='f8', shape=(shape[0],))
        self.stdev = np.ndarray(dtype='f8', shape=(shape[0],))


    def fit(self, data : np.ndarray) -> None:

        if self.log_transform:

            data = np.log(data)

        self.mean = np.nanmean(data, axis=1).reshape((-1, 1))
        self.stdev = np.nanstd(data, axis=1).reshape((-1, 1))


    def transform(self, data : np.ndarray) -> np.ndarray:

        if self.log_transform:

            data = np.log(data)

        return (data - self.mean) / self.stdev


class MedianNormalization(Normalization):

    def __init__(self, log_transform : bool = False, shape : tuple = None):

        self.log_transform = log_transform
        self.median = np.ndarray(dtype='f8', shape=(shape[0],))
        self.stdev = np.ndarray(dtype='f8', shape=(shape[0],))


    def fit(self, data : np.ndarray) -> None:

        if self.log_transform:

            data = np.log(data)

        self.median = np.nanmedian(data, axis=1).reshape((-1, 1))
        self.stdev = np.nanstd(data, axis=1).reshape((-1, 1))


    def transform(self, data : np.ndarray) -> np.ndarray:

        if self.log_transform:

            data = np.log(data)

        return (data - self.median) / self.stdev