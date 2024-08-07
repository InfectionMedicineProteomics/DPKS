"""**impute quantitative matrices**, supports multiple methods as specified below"""

import numpy as np

from sklearn.impute import KNNImputer


class ImputerMethod:
    """the base class"""

    def __init__(self) -> None:
        """init"""

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit the transform"""

        pass


class NeighborhoodImputer(ImputerMethod):
    def __init__(self, n_neighbors: int = 5, weights: str = "distance") -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)

        return imputer.fit_transform(X)


class UniformPercentileImputer(ImputerMethod):
    """uniform percentile imputer"""

    def __init__(self, percentile: float) -> None:
        """init"""
        self.percentile = percentile

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit the transform"""
        X = np.nan_to_num(X)  # Replace NaN with 0
        X_no_zero = X[X != 0]
        minvalue = X_no_zero.min()
        maxvalue = X_no_zero.max() * self.percentile + minvalue

        mask = X == 0
        c = np.count_nonzero(mask)
        nums = np.random.uniform(minvalue, maxvalue, c)
        X[mask] = nums

        return X


class ConstantImputer(ImputerMethod):
    """uniform percentile imputer"""

    def __init__(self, constant: float) -> None:
        """init"""
        self.constant = constant

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit the transform"""
        X = np.nan_to_num(X, nan=self.constant)  # Replace NaN with constant
        return X


class UniformRangeImputer(ImputerMethod):
    """uniform imputer
    >>> from dpks.quant_matrix import QuantMatrix
    >>> quant_matrix = QuantMatrix( quantification_file="tests/input_files/minimal_matrix_missing.tsv", design_matrix_file="tests/input_files/minimal_design_matrix.tsv")
    >>> uniform_imputed_df = quant_matrix.impute(method="uniform", minvalue=1, maxvalue=2).to_df()[["PeptideSequence", "SAMPLE_1.osw", "SAMPLE_2.osw", "SAMPLE_3.osw"]]
    >>> uniform_imputed_df
        PeptideSequence  SAMPLE_1.osw  SAMPLE_2.osw  SAMPLE_3.osw
    0         EFMEEVIQR       69900.3           1.0      403947.0
    1  SSSGTPDLPVLLTDLK      115684.0      132524.0      217962.0
    2            PEPTIK           1.0       59295.7       24536.4
    """

    def __init__(self, minvalue: int, maxvalue: int) -> None:
        """init"""
        self.minvalue = minvalue
        self.maxvalue = maxvalue

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit the transform"""
        X = np.nan_to_num(X)  # Replace NaN with 0
        mask = X == 0
        c = np.count_nonzero(mask)
        nums = np.random.uniform(self.minvalue, self.maxvalue, c)

        X[mask] = nums

        return X
