from typing import Tuple, List

import networkx as nx  # type: ignore
import numpy as np
import pandas as pd# type: ignore

from typing import TYPE_CHECKING

if TYPE_CHECKING:

    from .quant_matrix import QuantMatrix

class TopN:

    top_n: int
    protein_nodes: List[str]

    def __init__(self, top_n: int = 1):

        self.top_n = top_n
        self.num_proteins = 0
        self.num_samples = 0
        self.protein_nodes = []

    def quantify(self, quantitative_data : QuantMatrix) -> pd.DataFrame:

        protein_quantifications = dict()

        for protein in quantitative_data.proteins:

            protein_group = quantitative_data.quantitative_data[
                quantitative_data.quantitative_data.obs["Protein"] == protein
            ]

            protein_quantification = self.quantify_protein(protein_group.X)

            protein_quantifications[protein] = protein_quantification

        proteins = pd.DataFrame(protein_quantifications)

        proteins = proteins.T.copy()

        proteins.columns = list(
            quantitative_data.quantitative_data.var["sample"]
        )

        return proteins.reset_index().rename(
            columns={"index": "Protein"}
        )

    def quantify_protein(self, grouped_protein: np.ndarray) -> np.ndarray:

        grouped_protein = np.nan_to_num(grouped_protein, nan=0.0)

        sort_indices = np.argsort(grouped_protein, axis=0)[::-1]

        sorted_precursors = np.take_along_axis(grouped_protein, sort_indices, axis=0)

        protein_quantification: np.ndarray = np.sum(sorted_precursors[: self.top_n], axis=0)

        return protein_quantification
