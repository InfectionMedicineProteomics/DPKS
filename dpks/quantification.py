from abc import (
    ABC,
    abstractmethod
)
from typing import Tuple

import numpy as np

from enum import Enum


class ProteinQuantificationMethod(Enum):
    TOP_N_PRECURSORS = 1


class ProteinQuantification(ABC):

    @abstractmethod
    def fit(self, data) -> None:

        raise NotImplementedError

    @abstractmethod
    def quantify(self) -> np.ndarray:

        raise NotImplementedError


class TopNPrecursors(ProteinQuantification):

    top_n : int

    def __init__(self, top_n : int = 1):

        self.top_n = top_n

    def fit(self, quant_matrix):

        self.quant_matrix = quant_matrix

    def quantify(self) -> Tuple[np.ndarray, np.ndarray]:

        num_proteins = 0

        protein_nodes = []

        for node, node_data in self.quant_matrix.quant_graph.nodes(data=True):

            if node_data["bipartite"] == "protein":

                num_proteins += 1

                protein_nodes.append(node)

        protein_matrix = np.zeros(
            shape=(num_proteins, self.quant_matrix.num_samples),
            dtype='f8'
        )

        for protein_index, protein_node in enumerate(protein_nodes):

            precursors = []

            for precursor_node in self.quant_matrix.quant_graph.neighbors(protein_node):

                precursor_data = self.quant_matrix.quant_graph.nodes(data=True)[precursor_node]['data']

                precursors.append(precursor_data)


            local_protein_matrix = np.zeros(
                shape=(len(precursors), self.quant_matrix.num_samples),
                dtype='f8'
            )

            for local_index, precursor in enumerate(precursors):

                precursor_index = precursor.index

                for sample_index in range(self.quant_matrix.num_samples):

                    local_protein_matrix[local_index, sample_index] = self.quant_matrix.matrix[precursor_index, sample_index]


            if self.top_n == 1:

                protein_quantification = np.nanmax(local_protein_matrix, axis=0)


            for sample_index in range(self.quant_matrix.num_samples):

                protein_matrix[protein_index, sample_index] = protein_quantification[sample_index]


        return np.asarray(protein_nodes, dtype=str), protein_matrix