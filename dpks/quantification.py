from typing import List, Any, TYPE_CHECKING, Tuple, Union
from itertools import combinations, repeat
from multiprocessing import Pool

import warnings

import numba
import numpy as np
import pandas as pd  # type: ignore
from numba import njit, jit
from numpy.linalg import LinAlgError
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class TopN:

    top_n: int
    protein_nodes: List[str]

    def __init__(self, top_n: int = 1):

        self.top_n = top_n
        self.num_proteins = 0
        self.num_samples = 0
        self.protein_nodes = []

    def quantify(self, quantitative_data: QuantMatrix) -> pd.DataFrame:

        protein_quantifications = dict()

        for protein in quantitative_data.proteins:

            protein_group = quantitative_data.quantitative_data[
                quantitative_data.quantitative_data.obs["Protein"] == protein
            ]

            protein_quantification = self.quantify_protein(protein_group.X)

            protein_quantifications[protein] = protein_quantification

        proteins = pd.DataFrame(protein_quantifications)

        proteins = proteins.T.copy()

        proteins.columns = list(quantitative_data.quantitative_data.var["sample"])

        return proteins.reset_index().rename(columns={"index": "Protein"})

    def quantify_protein(self, grouped_protein: np.ndarray) -> np.ndarray:

        grouped_protein = np.nan_to_num(grouped_protein, nan=0.0)

        sort_indices = np.argsort(grouped_protein, axis=0)[::-1]

        sorted_precursors = np.take_along_axis(grouped_protein, sort_indices, axis=0)

        protein_quantification: np.ndarray = np.sum(
            sorted_precursors[: self.top_n], axis=0
        )

        return protein_quantification


@njit
def get_ratios(quantitative_data, sample_combinations, min_ratios):

    num_samples = quantitative_data.shape[1]

    ratios = np.empty(
        (num_samples, num_samples),
        dtype=np.float64
    )

    ratios[:] = np.nan

    for combination in sample_combinations:

        sample_a = combination[0]
        sample_b = combination[1]

        ratio =  - quantitative_data[:, sample_a] + quantitative_data[:, sample_b]

        non_nan = np.sum(~np.isnan(ratio))

        if non_nan >= min_ratios:

            ratio_median = np.nanmedian(ratio)

        else:

            ratio_median = np.nan

        ratios[sample_b, sample_a] = ratio_median

    return ratios

@njit
def solve_profile(X, ratios, sample_combinations):

    if np.all(np.isnan(X)):

        results = np.zeros((X.shape[1]))

    else:

        num_samples = X.shape[1]

        A = np.zeros((num_samples + 1, num_samples + 1))
        b = np.zeros((num_samples + 1,))

        for sample_combination in sample_combinations:

            i = sample_combination[0]
            j = sample_combination[1]

            A[i][j] = -1.0
            A[j][i] = -1.0
            A[i][i] += 1.0
            A[j][j] += 1.0

            ratio = ratios[j, i]

            if not np.isnan(ratio):
                b[i] -= ratio
                b[j] += ratio

        formatted_a = 2.0 * A
        formatted_a[:num_samples, num_samples] = 1
        formatted_a[num_samples, :num_samples] = 1

        formatted_b = 2.0 * b

        sample_mean = np.nanmean(X)

        if np.isnan(sample_mean):

            sample_mean = 0.0

        formatted_b[num_samples] = sample_mean * num_samples

        nan_idx = np.argwhere(np.isnan(b))

        for nan_value in nan_idx:

            formatted_b[nan_value] = 0.0

        results = np.linalg.lstsq(formatted_a, formatted_b, -1.0)[0][:X.shape[1]]

    results[results == 0.0] = np.nan

    return results


def quantify_groups(groupings, sample_combinations, group_ids, minimum_ratios):

    num_groups = len(group_ids)

    results = numba.typed.List()

    for group_idx in range(num_groups):

        grouping = groupings[group_idx]

        ratios = get_ratios(groupings[group_idx], sample_combinations, minimum_ratios)

        quantified_group = solve_profile(groupings[group_idx], ratios, sample_combinations)

        results.append(
            (group_ids[group_idx], quantified_group)
        )

    return results


class MaxLFQ:

    minimum_ratios: int
    level: str
    threads: int

    def __init__(self, minimum_ratios:int = 1, level: str = "protein", threads: int = 1):

        self.minimum_ratios = minimum_ratios
        self.level = level
        self.threads = threads

    def quantify(self, quantitative_data: QuantMatrix) -> pd.DataFrame:

        group_ids: List[str] = []
        key: str = ""

        if self.level == "protein":

            group_ids = quantitative_data.proteins
            key = "Protein"

        groupings = []

        for group_id in group_ids:

            groupings.append(
                quantitative_data.quantitative_data[
                    quantitative_data.quantitative_data.obs[key] == group_id
                ].X.copy()
            )

        column_idx = np.arange(0, groupings[0].shape[1])

        sample_combinations = np.array(list(combinations(column_idx, 2)))

        quantified_groups = quantify_groups(groupings, sample_combinations, group_ids, self.minimum_ratios)

        return self._format_result_df(quantified_groups, quantitative_data.sample_annotations["sample"].values)

    def _format_result_df(self, quantified_groups: List[Tuple[str, Union[np.ndarray, np.ndarray], bool]], samples: np.ndarray) -> pd.DataFrame:

        rows = []

        if self.level == "protein":

            for result in quantified_groups:

                result_id = result[0]
                result_data = result[1]

                row = {
                    "Protein": result_id,
                }


                for idx, sample in enumerate(samples):
                    row[sample] = result_data[idx]


                rows.append(row)

        return pd.DataFrame(rows)
