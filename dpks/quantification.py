from typing import List, Any, TYPE_CHECKING, Tuple, Union
from itertools import combinations, repeat
from multiprocessing import Pool

import warnings

import numpy as np
import pandas as pd  # type: ignore
from numba import njit
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

    num_combos = sample_combinations.shape[0]

    for combination in range(num_combos):

        sample_a = sample_combinations[combination][0]
        sample_b = sample_combinations[combination][1]

        ratio = quantitative_data[:, sample_a] / quantitative_data[:, sample_b]

        non_nan = np.sum(~np.isnan(ratio))

        if non_nan >= min_ratios:

            ratio_median = np.nanmedian(ratio)

        else:

            ratio_median = np.nan

        ratios[sample_b, sample_a] = ratio_median

    return ratios


@njit
def ss_loss(normalizations, ratios):

    estimates = np.repeat(normalizations, len(normalizations)).reshape((len(normalizations), len(normalizations))).transpose()

    loss = (np.log(ratios) - np.log(estimates.T) + np.log(estimates)) ** 2

    return np.nansum(loss)


def minimize_ratios(ratios, bounds, x0, method):

    options = {}

    if method == "Powell":

        options = {
            'disp': 0,
            'maxiter':int(1e6)
        }

    elif method == "L-BFGS-B":

        options = {
            'disp': 0,
            'maxiter':int(1e6),
            'maxfun':int(ratios.shape[0]*2e4),
            'eps': 1e-06
        }

    minimize_results = minimize(
        ss_loss,
        args = ratios,
        x0 = x0.ravel(),
        bounds = bounds,
        method=method,
        options=options
    )

    minimize_results.x = minimize_results.x / np.max(minimize_results.x)

    return minimize_results


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

        with warnings.catch_warnings():

            warnings.filterwarnings("ignore", category=RuntimeWarning)

            with Pool(self.threads) as p:

                quantified_groups = p.starmap(
                    self.quantify_protein,
                    zip(groupings, repeat(sample_combinations), group_ids)
                )

        return self._format_result_df(quantified_groups, quantitative_data.sample_annotations["sample"].values)

    def _format_result_df(self, quantified_groups: List[Tuple[str, Union[np.ndarray, np.ndarray], bool]], samples: np.ndarray) -> pd.DataFrame:

        rows = []

        if self.level == "protein":

            for result in quantified_groups:

                result_id = result[0]
                result_data = result[1]
                result_success = result[2]

                row = {
                    "Protein": result_id,
                }

                if result_success:

                    for idx, sample in enumerate(samples):
                        row[sample] = result_data[idx]

                else:

                    for idx, sample in enumerate(samples):
                        row[sample] = np.nan

                rows.append(row)

        return pd.DataFrame(rows)

    def quantify_protein(self, grouping: np.ndarray, sample_combinations: np.ndarray, group_id: str) -> Tuple[str, np.ndarray, bool]:

        ratios = get_ratios(grouping, sample_combinations, 1)

        num_samples = ratios.shape[1]

        x0 = np.ones(num_samples)

        bounds = [(min(np.nanmin(ratios), 1 / np.nanmax(ratios)), 1) for _ in x0]

        minimize_results = minimize_ratios(ratios, bounds, x0, "L-BFGS-B")

        if not minimize_results.success:

            minimize_results = minimize_ratios(ratios, bounds, x0, "Powell")

            if not minimize_results.success:
                minimize_results.x = np.zeros((num_samples,))

                print(f"Failed to determin profile for grouping {group_id}")

        sample_sums = np.nanmedian(grouping, axis=0)

        total_sum = np.nansum(sample_sums)

        corrected_values = total_sum * minimize_results.x

        sample_sums[sample_sums == 0] = np.nan

        nan_args = np.argwhere(np.isnan(sample_sums))

        profile = corrected_values * total_sum / np.sum(corrected_values)

        profile[nan_args] = np.nan

        return group_id, profile, minimize_results.success