"""**normalizes quantitative matrices**, supports multiple methods as specified below"""
from typing import TYPE_CHECKING, Any, List

import numpy as np
import pandas as pd  # type: ignore

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class NormalizationMethod:

    def __init__(self) -> None:

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        pass


class Log2Normalization(NormalizationMethod):

    def __init__(self) -> None:

        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        return np.array(np.log2(X), dtype=np.float64)


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
    minimum_data_points: int
    stride: float
    use_overlapping_windows: bool
    rt_unit: str

    def __init__(
        self,
        base_method: NormalizationMethod,
        stride: float = 1.0,
        minimum_data_points: int = 250,
        use_overlapping_windows: bool = False,
        rt_unit: str = "minute",
    ):

        self.base_method = base_method
        self.stride = stride
        self.minimum_data_points = minimum_data_points
        self.use_overlapping_windows = use_overlapping_windows
        self.rt_unit = rt_unit

    def build_rt_windows(self, rts: np.ndarray) -> List[np.ndarray]:

        if self.rt_unit == "seconds":

            rts = rts / 60.0

        rt_min = np.min(rts)
        rt_max = np.max(rts)

        rt_bins = np.arange(rt_min, rt_max, self.stride)

        rt_indices = np.digitize(rts, rt_bins) - 1

        bin_indices = []

        for rt_idx in range(len(rt_bins)):

            indices = np.argwhere(rt_indices == rt_idx)

            if indices.reshape(-1).shape[0] < self.minimum_data_points:

                expanded_indices = []

                expanded_indices.append(indices)

                for expand_index in range(1, len(rt_bins)):

                    lower_bound_idx = rt_idx - expand_index

                    upper_bound_idx = rt_idx + expand_index

                    if lower_bound_idx < 0:

                        expanded_indices.append(
                            np.argwhere(rt_indices == upper_bound_idx)
                        )

                    elif expand_index >= len(rt_bins):

                        expanded_indices.append(
                            np.argwhere(rt_indices == lower_bound_idx)
                        )

                    else:

                        expanded_indices.append(
                            np.argwhere(rt_indices == lower_bound_idx)
                        )

                        expanded_indices.append(
                            np.argwhere(rt_indices == upper_bound_idx)
                        )

                    if (
                        np.concatenate(expanded_indices).reshape(-1).shape[0]
                        >= self.minimum_data_points
                    ):
                        break

                bin_indices.append(np.concatenate(expanded_indices).reshape(-1))

            else:

                bin_indices.append(indices.reshape(-1))

        return bin_indices

    def overlap_rt_windows(self, rt_windows: List[np.ndarray]) -> List[np.ndarray]:

        new_rt_windows = []

        for rt_idx, rt_window in enumerate(rt_windows):

            if rt_idx == 0:

                new_rt_window = np.concatenate(
                    [
                        rt_window,
                        rt_windows[rt_idx + 1],
                    ]
                )

            elif rt_idx == len(rt_windows) - 1:

                new_rt_window = np.concatenate(
                    [
                        rt_window,
                        rt_windows[rt_idx - 1],
                    ]
                )

            else:

                new_rt_window = np.concatenate(
                    [
                        rt_windows[rt_idx - 1],
                        rt_window,
                        rt_windows[rt_idx + 1],
                    ]
                )

            new_rt_windows.append(new_rt_window)

        return new_rt_windows

    def fit_transform(self, quantitative_data: QuantMatrix) -> np.ndarray:

        normalized_slices = []

        rt_windows = self.build_rt_windows(
            rts=quantitative_data.quantitative_data.obs["RT"]
        )

        if self.use_overlapping_windows:

            rt_windows = self.overlap_rt_windows(rt_windows=rt_windows)

        for rt_window in rt_windows:

            normalized_slice = pd.DataFrame(
                self.base_method.fit_transform(
                    quantitative_data.quantitative_data[rt_window, :].X.copy()
                )
            ).set_index(pd.Index(rt_window))

            normalized_slices.append(normalized_slice)

        joined_slices = pd.concat(normalized_slices)

        joined_slices = joined_slices.reset_index().groupby("index", sort=False).mean()

        return_array: np.ndarray = joined_slices.to_numpy(dtype=np.float64)

        return return_array
