from __future__ import annotations

from typing import Union, List

import numpy as np
import pandas as pd  # type: ignore
import anndata as ad  # type: ignore

from dpks.normalization import TicNormalization, MedianNormalization, MeanNormalization, Log2Normalization, \
    NormalizationMethod, RTSlidingWindowNormalization
from dpks.quantification import TopN
from dpks.differential_testing import DifferentialTest


class QuantMatrix:

    quantification_file_path: Union[str, pd.DataFrame]
    design_matrix_file: Union[str, pd.DataFrame]
    num_rows: int
    num_samples: int
    quantitative_data: ad.AnnData

    def __init__(
        self,
        quantification_file: Union[str, pd.DataFrame],
        design_matrix_file: Union[str, pd.DataFrame],
        build_quant_graph: bool = False,
    ):

        if isinstance(design_matrix_file, str):

            design_matrix_file = pd.read_csv(design_matrix_file, sep="\t")

            design_matrix_file.columns = map(str.lower, design_matrix_file.columns)

        if isinstance(quantification_file, str):

            quantification_file = pd.read_csv(quantification_file, sep="\t")

        self.num_samples = len(design_matrix_file)
        self.num_rows = len(quantification_file)

        rt_column = ""

        if "RT" in quantification_file:

            rt_column = "RT"

        elif "RetentionTime" in quantification_file:

            rt_column = "RetentionTime"

        if rt_column:

            quantification_file.sort_values(rt_column, inplace=True)

        quantitative_data = (
            quantification_file[list(design_matrix_file["sample"])]
            .copy()
            .set_index(np.arange(self.num_rows, dtype=int).astype(str))
        )

        row_obs = quantification_file.drop(
            list(design_matrix_file["sample"]), axis=1
        ).set_index(np.arange(self.num_rows, dtype=int).astype(str))

        self.quantitative_data = ad.AnnData(
            quantitative_data,
            obs=row_obs,
            var=design_matrix_file.copy().set_index(design_matrix_file["sample"]),
            dtype=np.float64,
        )

        if build_quant_graph:

            pass

    @property
    def proteins(self) -> List[str]:

        return list(self.quantitative_data.obs["Protein"].unique())

    @property
    def precursors(self) -> List[str]:

        self.row_annotations["PrecursorId"] = (
            self.row_annotations["PeptideSequence"]
            + "_"
            + self.row_annotations["Charge"].astype(str)
        )

        return list(self.row_annotations["PrecursorId"].unique())

    @property
    def sample_annotations(self) -> pd.DataFrame:

        return self.quantitative_data.var

    @property
    def row_annotations(self) -> pd.DataFrame:

        return self.quantitative_data.obs

    @row_annotations.setter
    def row_annotations(self, value: pd.DataFrame) -> None:

        self.quantitative_data.obs = value

    def get_samples(self, group: int) -> List[str]:

        return list(
            self.sample_annotations[self.sample_annotations["group"] == group]["sample"]
        )

    def filter(
        self,
        peptide_q_value: float = 0.01,
        protein_q_value: float = 0.01,
        remove_decoys: bool = True,
        remove_contaminants: bool = True,
    ) -> QuantMatrix:

        filtered_data = self.quantitative_data[
            (self.quantitative_data.obs["PeptideQValue"] <= peptide_q_value)
            & (self.quantitative_data.obs["ProteinQValue"] <= protein_q_value)
        ].copy()

        if remove_decoys:

            filtered_data = filtered_data[filtered_data.obs["Decoy"] == 0].copy()

        if remove_contaminants:

            filtered_data = filtered_data[
                ~filtered_data.obs["Protein"].str.contains("contam")
            ].copy()

        self.quantitative_data = filtered_data

        return self

    def normalize(self,
                  method: str,
                  log_transform: bool = True,
                  use_rt_sliding_window_filter: bool = False,
                  **kwargs: int) -> QuantMatrix:

        base_method: NormalizationMethod = NormalizationMethod()

        if method == "tic":

            base_method = TicNormalization()

        elif method == "median":

            base_method = MedianNormalization()

        elif method == "mean":

            base_method = MeanNormalization()

        if use_rt_sliding_window_filter:

            rt_window_normalization = RTSlidingWindowNormalization(
                base_method=base_method,
                window_length=kwargs["window_length"],
                stride=kwargs["stride"]
            )

            self.quantitative_data.X = rt_window_normalization.fit_transform(
                self
            )

        else:

            self.quantitative_data.X = base_method.fit_transform(
                self.quantitative_data.X
            )

        if log_transform:

            self.quantitative_data.X = Log2Normalization().fit_transform(
                self.quantitative_data.X
            )

        return self

    def quantify(
        self, method: str, resolve_protein_groups: bool = False, **kwargs: int
    ) -> QuantMatrix:

        if resolve_protein_groups:

            pass

        if method == "top_n":

            quantifications = TopN(top_n=kwargs["top_n"]).quantify(self)

            design_matrix = self.quantitative_data.var

            protein_quantifications = QuantMatrix(
                quantifications, design_matrix_file=design_matrix
            )

        return protein_quantifications

    def to_df(self) -> pd.DataFrame:

        merged = pd.concat(
            [self.quantitative_data.obs, self.quantitative_data.to_df()], axis=1
        )

        return merged

    def compare_groups(
        self,
        method: str,
        group_a: int,
        group_b: int,
        min_samples_per_group: int = 2,
        level: str = "protein",
        multiple_testing_correction_method: str = "fdr_tsbh",
    ) -> QuantMatrix:

        differential_test = DifferentialTest(
            method,
            group_a,
            group_b,
            min_samples_per_group,
            level,
            multiple_testing_correction_method,
        )

        compared_data = differential_test.test(self)

        self.row_annotations = compared_data.row_annotations.copy()

        return self

    def impute(self) -> None:

        pass

    def outlier_detection(self) -> None:

        pass

    def flag_bad_runs(self) -> None:

        pass

    def write(self, file_path: str = "") -> None:

        self.to_df().to_csv(file_path, sep="\t", index=False)
