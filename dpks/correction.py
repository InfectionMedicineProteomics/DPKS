
import warnings

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd  # type: ignore
from inmoose.pycombat import pycombat_norm
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class CorrectionMethod:
    def __init__(self) -> None:
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass


class BatchCombat(CorrectionMethod):

    def __init__(self) -> None:
        pass

    def fit_transform(self, X: np.ndarray, batches) -> np.ndarray:
        
        le = LabelEncoder()
        batch_indices = le.fit_transform(batches)

        X_nan_to_num = np.nan_to_num(X, nan=0)
        corrected_data = pycombat_norm(X_nan_to_num, batch=batch_indices)

        return corrected_data