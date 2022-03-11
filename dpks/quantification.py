from abc import ABC, abstractmethod
from typing import Tuple, List

import networkx as nx  # type: ignore
import numpy as np  # type: ignore

from enum import Enum

import pandas as pd
from sklearn.base import TransformerMixin


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


class TopN:

    top_n: int
    protein_nodes: List[str]

    def __init__(self, top_n: int = 1):

        self.top_n = top_n
        self.num_proteins = 0
        self.num_samples = 0
        self.protein_nodes = []


    def quantify(self, quantitative_data, y=None, **fit_params):

        protein_quantifications = dict()

        for protein in quantitative_data.proteins:

            protein_group = quantitative_data.quantitative_data[
                quantitative_data.quantitative_data.obs["Protein"] == protein
            ]

            protein_quantification = self.quantify_protein(protein_group.X)

            protein_quantifications[protein] = protein_quantification

        protein_quantifications = pd.DataFrame(
            protein_quantifications
        )

        protein_quantifications = protein_quantifications.T.copy()

        protein_quantifications.columns = list(quantitative_data.quantitative_data.var["sample"])

        return protein_quantifications.reset_index().rename(columns={"index": "Protein"})

    def quantify_protein(self, grouped_protein: np.ndarray) -> np.ndarray:

        grouped_protein = np.nan_to_num(grouped_protein, nan=0.0)

        sort_indices = np.argsort(grouped_protein, axis=0)[::-1]

        sorted_precursors = np.take_along_axis(grouped_protein, sort_indices, axis=0)

        protein_quantification = np.sum(sorted_precursors[: self.top_n], axis=0)

        return protein_quantification
