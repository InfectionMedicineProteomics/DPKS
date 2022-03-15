"""**normalizes quantitative matrices**, supports multiple methods as specified below"""
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any

class NormalizationMethod:

    def __init__(self):

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        pass


class Log2Normalization(NormalizationMethod):

    def __init__(self):

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        return np.log2(X)


class TicNormalization(NormalizationMethod):
    def __init__(self) -> None:

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        sample_sums = np.nansum(X, axis=0)

        median_signal = np.nanmedian(sample_sums)

        normalized_signal: np.ndarray = (X / sample_sums[None, :]) * median_signal

        return normalized_signal


class MedianNormalization(NormalizationMethod):
    def __init__(self) -> None:

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        sample_medians = np.nanmedian(X, axis=0)

        mean_sample_median = np.mean(sample_medians)

        normalized_signal: np.ndarray = (
            X / sample_medians[None, :]
        ) * mean_sample_median

        return normalized_signal


class MeanNormalization(NormalizationMethod):
    """normalize the quantitative data using **mean**

    The input data is found in the DPKS git reposotory:

    - The minimal design matrix: `minimal_design_matrix.tsv`_. -  contains the experimental design
    - The minimal matrix: `minimal_matrix.tsv`_.  - contains the quantitative data

    .. _minimal_design_matrix.tsv: https://github.com/InfectionMedicineProteomics/DPKS/blob/doctest001/tests/input_files/minimal_design_matrix.tsv
    .. _minimal_matrix.tsv: https://github.com/InfectionMedicineProteomics/DPKS/blob/doctest001/tests/input_files/minimal_matrix.tsv

    The QuantMatrix instance is created by providing a quantitative matrix and a design matrix:

    >>> from dpks.quant_matrix import QuantMatrix
    >>> quant_matrix = QuantMatrix( quantification_file="tests/input_files/minimal_matrix.tsv", design_matrix_file="tests/input_files/minimal_design_matrix.tsv")
    >>> quant_matrix.to_df()
        PeptideSequence  Charge  Decoy Protein  RetentionTime  PeptideQValue  ProteinQValue  SAMPLE_1.osw  SAMPLE_2.osw  SAMPLE_3.osw
    0            PEPTIK       4      0  P00352        5736.15       0.000008       0.000117       29566.2       59295.7       24536.4
    1         EFMEEVIQR       2      0  P04275        3155.50       0.000009       0.000117       69900.3      195571.0      403947.0
    2  SSSGTPDLPVLLTDLK       2      0  P00352        5386.69       0.000008       0.000116      115684.0      132524.0      217962.0

    Applying the normalize method and specifying the method:

    >>> norm_quant_matrix = quant_matrix.normalize(method="mean")
    >>> norm_quant_matrix.to_df()[["PeptideSequence", "SAMPLE_1.osw", "SAMPLE_2.osw", "SAMPLE_3.osw"]]
        PeptideSequence  SAMPLE_1.osw  SAMPLE_2.osw  SAMPLE_3.osw
    0            PEPTIK     15.804039     15.959574     13.947831
    1         EFMEEVIQR     17.045388     17.681267     17.989002
    2  SSSGTPDLPVLLTDLK     17.772207     17.119828     17.098912

    """

    def __init__(self) -> None:

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        sample_means = np.nanmean(X, axis=0)

        mean_sample_means = np.mean(sample_means)

        normalized_signal: np.ndarray = (X / sample_means[None, :]) * mean_sample_means

        return normalized_signal


class RTSlidingWindowNormalization:

    base_method: NormalizationMethod
    window_length: int
    stride: int

    def __init__(self,
                 base_method: NormalizationMethod,
                 window_length: int = 25,
                 stride: int = 5):

        self.base_method = base_method
        self.window_length = window_length
        self.stride = stride

    def fit_transform(self, quantitative_data: QuantMatrix) -> QuantMatrix:

        for rt_window in sliding_window_view(quantitative_data.row_annotations.index, self.window_length)[::self.stride, :]:

            quantitative_data.quantitative_data[rt_window, :].X = self.base_method().fit_transform(
                quantitative_data.quantitative_data[rt_window, :].X
            )

        return quantitative_data
