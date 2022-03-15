"""**normalizes quantitative matrices**, supports multiple methods as specified below"""
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
    """normalize the quantitative data using **mean**

    The input data is found in the DPKS git reposotory:

    - The minimal design matrix: `minimal_design_matrix.tsv`_. -  contains the experimental design
    - The minimal matrix: `minimal_matrix.tsv`_.  - contains the quantitative data

    .. _minimal_design_matrix.tsv: https://github.com/InfectionMedicineProteomics/DPKS/blob/main/tests/input_files/design_matrix.tsv
    .. _minimal_matrix.tsv: https://github.com/InfectionMedicineProteomics/DPKS/blob/main/tests/input_files/pyprophet_baseline_matrix.tsv

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

        normalized_signal = np.log2(normalized_signal)

        return normalized_signal
