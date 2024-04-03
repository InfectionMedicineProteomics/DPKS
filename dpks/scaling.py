"""**scales quantitative matrices**, supports multiple methods as specified below

>>> from dpks.quant_matrix import QuantMatrix
>>> quant_matrix = QuantMatrix( quantification_file="tests/input_files/minimal_matrix.tsv", design_matrix_file="tests/input_files/minimal_design_matrix.tsv")
>>> quant_matrix.to_df()[["PeptideSequence", "SAMPLE_1.osw", "SAMPLE_2.osw", "SAMPLE_3.osw"]]
    PeptideSequence  SAMPLE_1.osw  SAMPLE_2.osw  SAMPLE_3.osw
0         EFMEEVIQR       69900.3      195571.0      403947.0
1  SSSGTPDLPVLLTDLK      115684.0      132524.0      217962.0
2            PEPTIK       29566.2       59295.7       24536.4

"""

from typing import TYPE_CHECKING, Any
import numpy as np

# import numpy.typing as npt  # not used yet

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class ScalingMethod:
    """the base class"""

    def __init__(self) -> None:
        """init"""

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """carry out the transform"""

        pass


class ZScoreScaling(ScalingMethod):
    """z-score scaling

    >>> from dpks.quant_matrix import QuantMatrix
    >>> quant_matrix = QuantMatrix( quantification_file="tests/input_files/minimal_matrix.tsv", design_matrix_file="tests/input_files/minimal_design_matrix.tsv")
    >>> zscore_scaled_df = quant_matrix.scale(method="zscore").to_df()[["PeptideSequence", "SAMPLE_1.osw", "SAMPLE_2.osw", "SAMPLE_3.osw"]]
    >>> zscore_scaled_df
        PeptideSequence  SAMPLE_1.osw  SAMPLE_2.osw  SAMPLE_3.osw
    0         EFMEEVIQR     -1.112361     -0.200119      1.312480
    1  SSSGTPDLPVLLTDLK     -0.886769     -0.510675      1.397444
    2            PEPTIK     -0.536779      1.401483     -0.864704
    >>> zscore_scaled_df.set_index("PeptideSequence").sum(axis=1)
    PeptideSequence
    EFMEEVIQR          -4.440892e-16
    SSSGTPDLPVLLTDLK    0.000000e+00
    PEPTIK              1.110223e-15
    dtype: float64

    """

    def __init__(self) -> None:
        """init"""

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """carry out the z-score normalization"""

        means = np.nanmean(X, axis=1)
        stddevs = np.nanstd(X, axis=1)

        return np.array((X - means[:, None]) / stddevs[:, None])


class MinMaxScaling(ScalingMethod):
    """min-max scaling

    Computes the min max scaling: (X - X_min) / (X_max - X_min)

    Resulting values are in range [0 1]
    """

    def __init__(self) -> None:
        """init"""

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """carry out the min-max normalization"""

        mins = np.nanmin(X, axis=1)
        maxes = np.nanmax(X, axis=1)

        return np.array((X - mins[:, None]) / (maxes[:, None] - mins[:, None]))


class AbsMaxScaling(ScalingMethod):
    """abs-max scaling

    Computes the abs-max scaling: X / X_max

    Resulting values are in range [-1 1]
    """

    def __init__(self) -> None:
        """init"""

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """carry out the min-max normalization"""

        maxes = np.nanmax(X, axis=1)

        return np.array(X / maxes[:, None])
