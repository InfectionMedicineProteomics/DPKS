from __future__ import annotations

from csv import DictReader
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
from dpks.quantification import ProteinQuantificationMethod, TopNPrecursors


class Protein:

    accession: str
    index: int

    def __init__(self, accession: str = "", index: int = -1):

        self.accession = accession
        self.index = index


class Precursor:

    peptide_sequence: str
    charge: int
    decoy: int
    retention_time: float
    index: int

    def __init__(
        self,
        peptide_sequence: str = "",
        charge: int = 0,
        decoy: int = 0,
        retention_time: float = 0.0,
        index: int = 0,
    ):

        self.peptide_sequence = peptide_sequence
        self.charge = charge
        self.decoy = decoy
        self.retention_time = retention_time
        self.index = index


class Fragment:

    pass


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

    def filter(self, peptide_q_value: float = 0.01, protein_q_value: float = 0.01, remove_decoys: bool = True):

        filtered_data = self.quantitative_data[
            (self.quantitative_data.obs["PeptideQValue"] <= peptide_q_value) &
            (self.quantitative_data.obs["ProteinQValue"] <= protein_q_value)
        ].copy()

        if remove_decoys:

            filtered_data = filtered_data[
                filtered_data.obs["Decoy"] == 0
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
        method: ProteinQuantificationMethod,
        top_n: int = 1,
        protein_grouping: str = "",
    ):

        if method.value == ProteinQuantificationMethod.TOP_N_PRECURSORS.value:

            quantification = TopNPrecursors(
                top_n=top_n, protein_grouping=protein_grouping
            )

        quantification.init(self.matrix, self.quant_graph)

        quantified_protein_ids, quantified_protein_data = quantification.quantify()

        return self._copy(
            num_samples=self.num_samples,
            num_quant_records=self.num_quant_records,
            quant_type=self.quant_type,
            design_matrix=self.design_matrix,
            matrix=quantified_protein_data,
            quant_graph=self.quant_graph,
            quant_record_index=quantified_protein_ids,
        )

    def _copy(
        self,
        num_samples: int = 0,
        num_quant_records: int = 0,
        quant_type: str = "",
        design_matrix: list = None,
        matrix: np.ndarray = None,
        quant_graph: nx.Graph = None,
        quant_record_index: np.ndarray = None,
    ):

        cloned = self.__class__
        obj = cloned(
            num_samples=num_samples,
            num_quant_records=num_quant_records,
            quant_type=quant_type,
            design_matrix=design_matrix,
            matrix=matrix,
            quant_record_index=quant_record_index,
            quant_graph=quant_graph,
        )

        return obj

    def differentiate_expression(self):

        pass


    def impute(self):

        pass

    def outlier_detection(self):

        pass

    def flag_bad_runs(self):

        pass

    def write(self, file_path: str = ""):

        pass

    def as_dataframe(self, level: str = ""):

        records = []

        if level == "protein":

            num_proteins: int = self.quant_record_index.shape[0]

            for protein_index in range(num_proteins):

                protein_id: str = self.quant_record_index[protein_index]

                record: dict = {
                    "Protein": protein_id,
                }

                for sample_index, sample_data in enumerate(self.design_matrix):

                    record[sample_data["name"]] = self.matrix[
                        protein_index, sample_index
                    ]

                records.append(record)

        if level == "precursor":

            pass

        if level == "peptide":

            pass

        if level == "fragment":

            pass

        return pd.DataFrame(records)

    @classmethod
    def from_csv(
        cls, file_path: str = "", design_matrix_path: str = "", quant_type: str = ""
    ) -> QuantMatrix:

        assert quant_type in ["fragment", "precursor", "peptide", "protein"]

        design_matrix = list()

        with open(design_matrix_path, "r") as design_matrix_path_h:

            csv_reader = DictReader(design_matrix_path_h)

            for record in csv_reader:

                design_matrix_record = {"name": record["Sample"]}

                if "Group" in record:

                    design_matrix_record["group"] = record["Group"]

                if "Batch" in record:

                    design_matrix_record["batch"] = record["Batch"]

                design_matrix.append(design_matrix_record)

        with open(file_path, "r") as quant_file_h:

            csv_reader = DictReader(quant_file_h)

            records = list(csv_reader)

            num_quant_records = len(records)

            quant_matrix = QuantMatrix(
                design_matrix=design_matrix,
                num_samples=len(design_matrix),
                num_quant_records=num_quant_records,
                quant_type=quant_type,
            )

            for index, record in enumerate(records):

                if quant_type == "precursor":

                    precursor_id = f"{record['PeptideSequence']}_{record['Charge']}"

                    quant_matrix.quant_record_index[index] = precursor_id

                    precursor = Precursor(
                        peptide_sequence=record["PeptideSequence"],
                        charge=record["Charge"],  # type: ignore
                        decoy=record["Decoy"],  # type: ignore
                        retention_time=record["RetentionTime"],  # type: ignore
                        index=index,
                    )

                    quant_matrix.quant_graph.add_nodes_from(
                        [precursor_id], data=precursor, bipartite="precursor"
                    )

                    protein_accessions = record["Protein"].split(";")

                    for protein_accession in protein_accessions:

                        protein = Protein(accession=protein_accession)

                        if protein_accession not in quant_matrix.quant_graph:

                            quant_matrix.quant_graph.add_nodes_from(
                                [protein_accession], data=protein, bipartite="protein"
                            )

                        quant_matrix.quant_graph.add_edges_from(
                            [(protein_accession, precursor_id)]
                        )

                    for sample_index, sample_info in enumerate(design_matrix):

                        sample_name = sample_info["name"]

                        intensity = record[sample_name]

                        quant_matrix.matrix[index, sample_index] = intensity

        return quant_matrix


def create_quant_matrix(
    file_path: str = "", design_matrix: str = "", quant_type: str = ""
) -> QuantMatrix:

    print("Parsing Quant Matrix and building data structure.")

    return QuantMatrix.from_csv(file_path, design_matrix, quant_type)
