from __future__ import annotations

from typing import Union

import networkx as nx  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import anndata as ad

from dpks.normalization import (
    TicNormalization,
    MedianNormalization,
    MeanNormalization
)
from dpks.quantification import TopN


class QuantMatrix:

    quantification_file_path: Union[str, pd.DataFrame]
    design_matrix_file: Union[str, pd.DataFrame]
    num_rows: int
    num_samples: int
    quantitative_data: ad.AnnData
    # matrix: np.ndarray
    # design_matrix: list[dict[str, str]]
    # data_sets: dict[str, np.ndarray]
    # num_samples: int
    # num_quant_records: int
    # quant_ids: np.ndarray

    def __init__(
        self,
        quantification_file: Union[str, pd.DataFrame],
        design_matrix_file: Union[str, pd.DataFrame],
        build_quant_graph: bool = False
    ):

        if isinstance(design_matrix_file, str):

            design_matrix_file = pd.read_csv(
                design_matrix_file,
                sep="\t"
            )

            design_matrix_file.columns = map(
                str.lower,
                design_matrix_file.columns
            )

        if isinstance(quantification_file, str):

            quantification_file = pd.read_csv(
                quantification_file,
                sep="\t"
            )

        self.num_samples = len(design_matrix_file)
        self.num_rows = len(quantification_file)

        quantitative_data = quantification_file[list(design_matrix_file["sample"])].copy().set_index(
            np.arange(
                self.num_rows,
                dtype=int
            ).astype(str)
        )

        row_obs = quantification_file.drop(
            list(design_matrix_file["sample"]),
            axis=1
        ).set_index(
            np.arange(
                self.num_rows,
                dtype=int
            ).astype(str)
        )

        self.quantitative_data = ad.AnnData(
            quantitative_data,
            obs=row_obs,
            var=design_matrix_file.copy().set_index(design_matrix_file["sample"]),
            dtype=np.float64
        )

        if build_quant_graph:

            pass

    @property
    def proteins(self):

        return list(self.quantitative_data.obs["Protein"].unique())

    def filter(self,
               peptide_q_value: float = 0.01,
               protein_q_value: float = 0.01,
               remove_decoys: bool = True,
               remove_contaminants: bool = True):

        filtered_data = self.quantitative_data[
            (self.quantitative_data.obs["PeptideQValue"] <= peptide_q_value) &
            (self.quantitative_data.obs["ProteinQValue"] <= protein_q_value)
        ].copy()

        if remove_decoys:

            filtered_data = filtered_data[
                filtered_data.obs["Decoy"] == 0
            ].copy()

        if remove_contaminants:

            filtered_data = filtered_data[
                ~filtered_data.obs["Protein"].str.contains("contam")
            ].copy()

        self.quantitative_data = filtered_data

        return self


    def normalize(self, method: str):

        if method == "tic":

            self.quantitative_data.X = TicNormalization().fit_transform(
                self.quantitative_data.X
            )

        elif method == "median":

            self.quantitative_data.X = MedianNormalization().fit_transform(
                self.quantitative_data.X
            )

        elif method == "mean":

            self.quantitative_data.X = MeanNormalization().fit_transform(
                self.quantitative_data.X
            )

        return self

    def quantify(
        self,
        method: str,
        resolve_protein_groups: bool = False,
        **kwargs
    ):

        if resolve_protein_groups:

            pass

        if method == "top_n":

            quantifications = TopN(top_n=kwargs['top_n']).quantify(
                self
            )

            design_matrix = self.quantitative_data.var

            protein_quantifications = QuantMatrix(
                quantifications,
                design_matrix_file=design_matrix
            )

            return protein_quantifications

    def to_df(self):

        merged = pd.concat(
            [self.quantitative_data.obs, self.quantitative_data.to_df()],
            axis=1
        )

        return merged

    def test_differential_expression(self):

        pass


    def impute(self):

        pass

    def outlier_detection(self):

        pass

    def flag_bad_runs(self):

        pass

    def write(self, file_path: str = ""):

        pass
