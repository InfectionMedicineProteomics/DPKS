"""**impute quantitative matrices**, supports multiple methods as specified below"""
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class ImputerMethod:
    """the base class"""

    def __init__(self) -> None:
        """init"""

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit the transform"""

        pass


class ImputerRandom(ImputerMethod):
    """z-score scaling

    >>> from dpks.quant_matrix import QuantMatrix
    >>> quant_matrix = QuantMatrix( quantification_file="tests/input_files/minimal_matrix_missing.tsv", design_matrix_file="tests/input_files/minimal_design_matrix.tsv")
    >>> random_imputed_df = quant_matrix.impute(method="random", minvalue=1, maxvalue=2).to_df()[["PeptideSequence", "SAMPLE_1.osw", "SAMPLE_2.osw", "SAMPLE_3.osw"]]
    >>> random_imputed_df
        PeptideSequence  SAMPLE_1.osw  SAMPLE_2.osw  SAMPLE_3.osw
    0         EFMEEVIQR       69900.3           1.0      403947.0
    1  SSSGTPDLPVLLTDLK      115684.0      132524.0      217962.0
    2            PEPTIK           1.0       59295.7       24536.4

    """

    def __init__(self, minvalue: int = 1, maxvalue: int = 100) -> None:
        """init"""
        self.minvalue = minvalue
        self.maxvalue = maxvalue

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit the transform"""

        mask = X == 0
        c = np.count_nonzero(mask)
        nums = np.random.randint(self.minvalue, self.maxvalue, c)

        X[mask] = nums

        return X
