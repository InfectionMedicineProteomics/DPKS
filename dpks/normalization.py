import numpy as np


class TicNormalization:
    def __init__(self) -> None:

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        sample_sums = np.nansum(X, axis=0)

        median_signal = np.nanmedian(sample_sums)

        normalized_signal: np.ndarray = (X / sample_sums[None, :]) * median_signal

        normalized_signal = np.log2(normalized_signal)

        return normalized_signal


class MedianNormalization:
    def __init__(self) -> None:

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        sample_medians = np.nanmedian(X, axis=0)

        mean_sample_median = np.mean(sample_medians)

        normalized_signal: np.ndarray = (
            X / sample_medians[None, :]
        ) * mean_sample_median

        normalized_signal = np.log2(normalized_signal)

        return normalized_signal


class MeanNormalization:
    def __init__(self) -> None:

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        sample_means = np.nanmean(X, axis=0)

        mean_sample_means = np.mean(sample_means)

        normalized_signal: np.ndarray = (X / sample_means[None, :]) * mean_sample_means

        normalized_signal = np.log2(normalized_signal)

        return normalized_signal
