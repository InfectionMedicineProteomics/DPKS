"""**normalizes quantitative matrices**, supports multiple methods as specified below"""
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
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
    stride: float

    def __init__(self,
                 base_method: NormalizationMethod,
                 window_length: int = 25,
                 stride: float = 1.0,
                 minimum_data_points: int = 100):

        self.base_method = base_method
        self.window_length = window_length
        self.stride = stride
        self.minimum_data_points = minimum_data_points

    def fit_transform(self, quantitative_data: QuantMatrix) -> np.ndarray:

        quantitative_data.quantitative_data.obs["RT"] = quantitative_data.quantitative_data.obs["RT"] / 60.0

        rt_min = quantitative_data.quantitative_data.obs["RT"].min()
        rt_max = quantitative_data.quantitative_data.obs["RT"].max()
        step_size_minutes = self.stride

        for window_start in np.arange(rt_min, rt_max, self.stride):

            rt_slice = quantitative_data.quantitative_data[
                (quantitative_data.quantitative_data.obs["RT"] >= window_start) &
                (quantitative_data.quantitative_data.obs["RT"] < window_start + step_size_minutes)
            ].obs.index

            if rt_slice.any():

                if len(rt_slice) < self.minimum_data_points:

                    remaining_data_points = self.minimum_data_points - len(rt_slice)

                    pick_before = np.floor(remaining_data_points / 2).astype(int)

                    pick_after = np.ceil(remaining_data_points / 2).astype(int)

                    number_before = len(
                        quantitative_data.quantitative_data[
                            quantitative_data.quantitative_data.obs["RT"] < window_start
                            ]
                    )

                    number_after = len(
                        quantitative_data.quantitative_data[
                            quantitative_data.quantitative_data.obs["RT"] >= window_start + step_size_minutes
                            ]
                    )

                    if pick_before > number_before:

                        diff = pick_before - number_before

                        pick_after = pick_after + diff

                        pick_before = pick_before - diff

                    elif pick_after > number_after:

                        diff = pick_after - number_after

                        pick_before = pick_before + diff

                        pick_after = pick_after - diff

                    start_index = int(rt_slice[0])
                    end_index = int(rt_slice[-1])

                    quantitative_data.quantitative_data[(start_index - pick_before):(end_index + pick_after), :].X = self.base_method.fit_transform(
                        quantitative_data.quantitative_data[(start_index - pick_before):(end_index + pick_after), :].X.copy()
                    )

                else:

                    quantitative_data.quantitative_data[
                        (quantitative_data.quantitative_data.obs["RT"] >= window_start) &
                        (quantitative_data.quantitative_data.obs["RT"] < window_start + step_size_minutes)
                    ].X = self.base_method.fit_transform(
                        quantitative_data.quantitative_data[
                            (quantitative_data.quantitative_data.obs["RT"] >= window_start) &
                            (quantitative_data.quantitative_data.obs["RT"] < window_start + step_size_minutes)
                        ].X.copy()
                    )


        # for rt_window in sliding_window_view(quantitative_data.row_annotations.index, self.window_length)[::self.stride, :]:
        #
        #     quantitative_data.quantitative_data[rt_window, :].X = self.base_method.fit_transform(
        #         quantitative_data.quantitative_data[rt_window, :].X.copy()
        #     )

            # normalized_slice = pd.DataFrame(
            #     self.base_method.fit_transform(
            #         quantitative_data.quantitative_data[rt_window, :].X.copy()
            #     )
            # ).set_index(pd.Index(rt_window))
            #
            # normalized_slices.append(normalized_slice)

        # joined_slices = pd.concat(normalized_slices)
        #
        # joined_slices = joined_slices.reset_index().groupby("index", sort=False).mean()

        return quantitative_data.quantitative_data.X
