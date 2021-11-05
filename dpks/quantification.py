from abc import ABC, abstractmethod
from typing import Tuple, List

import networkx as nx  # type: ignore
import numpy as np  # type: ignore

from enum import Enum


class ProteinQuantificationMethod(Enum):
    TOP_N_PRECURSORS = 1


class ProteinQuantification(ABC):
    @abstractmethod
    def init(self, quant_data: np.ndarray, protein_graph: nx.Graph) -> None:

        raise NotImplementedError

    @abstractmethod
    def quantify(self) -> Tuple[np.ndarray, np.ndarray]:

        raise NotImplementedError

    @abstractmethod
    def build_protein_group(self, protein_node: str) -> np.ndarray:

        raise NotImplementedError

    @abstractmethod
    def quantify_protein(self, group_protein: np.ndarray) -> np.ndarray:

        raise NotImplementedError


class TopNPrecursors(ProteinQuantification):

    top_n: int
    protein_nodes: List[str]

    def __init__(self, top_n: int = 1, protein_grouping: str = ""):

        self.top_n = top_n
        self.protein_grouping = protein_grouping
        self.num_proteins = 0
        self.num_samples = 0
        self.protein_nodes = []

    def init(self, quant_data: np.ndarray, protein_graph: nx.Graph):

        self.num_samples = quant_data.shape[1]

        for node, node_data in protein_graph.nodes(data=True):

            if node_data["bipartite"] == "protein":

                self.num_proteins += 1

                self.protein_nodes.append(node)

        self.protein_matrix = np.zeros(
            shape=(self.num_proteins, self.num_samples), dtype="f8"
        )

        self.quant_matrix = quant_data
        self.protein_graph = protein_graph

    def build_protein_group(self, protein_node: str) -> np.ndarray:

        precursors: list = []

        for precursor_node in self.protein_graph.neighbors(protein_node):

            precursor_data = self.protein_graph.nodes(data=True)[precursor_node]["data"]

            if self.protein_grouping == "proteotypic":

                num_mapped_proteins = len(
                    list(self.protein_graph.neighbors(precursor_node))
                )

                if num_mapped_proteins == 1:

                    precursors.append(precursor_data)

        local_protein_matrix = np.zeros(
            shape=(len(precursors), self.num_samples), dtype="f8"
        )

        for local_index, precursor in enumerate(precursors):

            precursor_index = precursor.index

            for sample_index in range(self.num_samples):
                local_protein_matrix[local_index, sample_index] = self.quant_matrix[
                    precursor_index, sample_index
                ]

        return local_protein_matrix

    def quantify(self) -> Tuple[np.ndarray, np.ndarray]:

        for protein_index, protein_node in enumerate(self.protein_nodes):

            grouped_protein = self.build_protein_group(protein_node)

            protein_quantification = self.quantify_protein(grouped_protein)

            for sample_index in range(self.num_samples):

                self.protein_matrix[
                    protein_index, sample_index
                ] = protein_quantification[sample_index]

        return np.asarray(self.protein_nodes, dtype=str), self.protein_matrix

    def quantify_protein(self, grouped_protein: np.ndarray) -> np.ndarray:

        grouped_protein = np.nan_to_num(grouped_protein, nan=0.0)

        sort_indices = np.argsort(grouped_protein, axis=0)[::-1]

        sorted_precursors = np.take_along_axis(grouped_protein, sort_indices, axis=0)

        protein_quantification = np.sum(sorted_precursors[: self.top_n], axis=0)

        return protein_quantification
