import numpy as np


class CorrectionMethod:
    def __init__(self) -> None:
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass


class MeanCorrection(CorrectionMethod):

    def __init__(self, reference_batch) -> None:
        self.reference_batch = reference_batch

    def fit_transform(
        self,
        X: np.ndarray,
        batches,
    ) -> np.ndarray:

        X = X.T
        unique_batches = np.unique(batches)
        X_normalized = np.zeros_like(X)

        reference_indices = np.where(batches == self.reference_batch)[0]
        reference_data = X[reference_indices]

        reference_mean = np.nanmean(reference_data, axis=0)
        reference_std = np.nanstd(reference_data, axis=0)

        for batch in unique_batches:
            batch_indices = np.where(batches == batch)[0]
            batch_data = X[batch_indices]

            batch_mean = np.nanmean(batch_data, axis=0)
            batch_std = np.nanstd(batch_data, axis=0)

            normalized_batch_data = (batch_data - batch_mean) * (
                reference_std / batch_std
            ) + reference_mean

            X_normalized[batch_indices] = normalized_batch_data

        return X_normalized.T
