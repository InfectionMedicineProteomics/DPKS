from typing import List, Any, TYPE_CHECKING, Tuple, Union

# from itertools import combinations, repeat  # not used yet
# from multiprocessing import Pool  # not used yet

#  import warnings  # not used yet

import numba  # type: ignore
import numpy as np

# import numpy.typing as npt  # not used yet
import pandas as pd  # type: ignore
from numba import njit, prange  # type: ignore

# from numpy.linalg import LinAlgError  # type: ignore

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class TopN:
    top_n: int

    def __init__(
        self, top_n: int = 1, level: str = "protein", summarization_method: str = "sum"
    ):
        self.top_n = top_n
        self.num_proteins = 0
        self.num_samples = 0

        self.level = level
        self.summarization_method = summarization_method

    def quantify(self, quant_matrix: QuantMatrix) -> pd.DataFrame:
        quantifications = dict()

        if self.level == "protein":
            key = "Protein"
            group_ids = quant_matrix.proteins

        elif self.level == "precursor":
            key = "PrecursorId"
            group_ids = quant_matrix.precursors

        elif self.level == "peptide":
            key = "PeptideSequence"
            group_ids = quant_matrix.peptides

        elif self.level in quant_matrix.quantitative_data.obs.columns:
            key = self.level
            group_ids = list(quant_matrix.quantitative_data.obs[key].unique())
        else:
            raise ValueError("The level is not valid.")

        for group_id in group_ids:
            group_data = quant_matrix.quantitative_data[
                quant_matrix.quantitative_data.obs[key] == group_id
            ]

            quantification = self.quantify_group(group_data.X)

            quantifications[group_id] = quantification

        results = pd.DataFrame(quantifications)

        results = results.T.copy()

        results.columns = list(quant_matrix.quantitative_data.var["sample"])

        return results.reset_index().rename(columns={"index": key})

    def quantify_group(self, group_data: np.ndarray) -> np.ndarray:
        group_data = np.nan_to_num(group_data, nan=0.0)

        sort_indices = np.argsort(group_data, axis=0)[::-1]

        sorted_precursors = np.take_along_axis(group_data, sort_indices, axis=0)

        if self.summarization_method == "sum":
            quantification: np.ndarray = np.sum(sorted_precursors[: self.top_n], axis=0)

        elif self.summarization_method == "mean":
            quantification: np.ndarray = np.mean(
                sorted_precursors[: self.top_n], axis=0
            )

        elif self.summarization_method == "median":
            quantification: np.ndarray = np.median(
                sorted_precursors[: self.top_n], axis=0
            )

        return quantification


@njit(nogil=True)
def get_ratios(quantitative_data, sample_combinations):
    num_samples = quantitative_data.shape[1]

    ratios = np.empty((num_samples, num_samples), dtype=np.float64)

    ratios[:] = np.nan

    for combination in sample_combinations:
        sample_a = combination[0]
        sample_b = combination[1]

        ratio = -quantitative_data[:, sample_a] + quantitative_data[:, sample_b]

        ratio_median = np.nanmedian(ratio)

        ratios[sample_b, sample_a] = ratio_median

    return ratios


@njit(nogil=True)
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

        results = np.linalg.lstsq(formatted_a, formatted_b, -1.0)[0][: X.shape[1]]

    results[results == 0.0] = np.nan

    return results


@njit(nogil=True)
def build_connection_graph(grouping):
    connected_sample_groups = numba.typed.Dict()

    connected_indices = numba.typed.List()

    sample_group_id = 0

    for sample_idx in range(grouping.shape[1]):
        if sample_idx not in connected_indices:
            sample_group = []

            for compared_sample_idx in range(grouping.shape[1]):
                comparison = grouping[:, sample_idx] - grouping[:, compared_sample_idx]

                if not np.isnan(comparison).all():
                    sample_group.append(compared_sample_idx)

                    connected_indices.append(compared_sample_idx)

            if len(sample_group) > 0:
                connected_sample_groups[sample_group_id] = np.array(sample_group)

                sample_group_id += 1

            else:
                connected_sample_groups[sample_group_id] = np.array([sample_idx])

                sample_group_id += 1

    return connected_sample_groups


@njit(nogil=True)
def build_combinations(subset):
    column_idx = np.arange(0, subset.shape[1])

    combos = []

    for i in column_idx:
        for j in range(i + 1, column_idx.shape[0]):
            combos.append([i, j])

    return np.array(combos)


@njit(nogil=True)
def mask_group(grouping):
    nan_groups = []

    for subgroup_idx in range(grouping.shape[0]):
        if not np.isnan(grouping[subgroup_idx, :]).all():
            nan_groups.append(subgroup_idx)

    grouping = grouping[np.array(nan_groups), :]

    return grouping


@njit(nogil=True)
def quantify_group(grouping, connected_graph):
    profile = np.zeros((grouping.shape[1]))

    for sample_group_id, graph in connected_graph.items():
        if graph.shape[0] == 1:
            subset = grouping[:, graph]

            if np.isnan(subset).all():
                profile[graph] = np.nan

            else:
                profile[graph] = np.nanmedian(subset)

        if graph.shape[0] > 1:
            subset = grouping[:, graph]

            sample_combinations = build_combinations(subset)

            ratios = get_ratios(subset, sample_combinations)

            solved_profile = solve_profile(subset, ratios, sample_combinations)

            for results_idx in range(solved_profile.shape[0]):
                profile[graph[results_idx]] = solved_profile[results_idx]

    return profile


@njit(parallel=True)
def quantify_groups(groupings, group_ids, minimum_subgroups):
    num_groups = len(group_ids)

    results = np.empty(shape=(num_groups, groupings[0].shape[1]))

    result_ids = ["" for _ in range(num_groups)]

    for group_idx in prange(num_groups):
        grouping = mask_group(groupings[group_idx])

        if grouping.shape[0] >= minimum_subgroups:
            connected_graph = build_connection_graph(grouping)

            profile = quantify_group(grouping, connected_graph)

        else:
            profile = np.zeros((grouping.shape[1]))
            profile[:] = np.nan

        for sample_idx in range(profile.shape[0]):
            results[group_idx, sample_idx] = profile[sample_idx]

        result_ids[group_idx] = group_ids[group_idx]

    return_results = numba.typed.List()

    for group_idx in range(num_groups):
        return_results.append((result_ids[group_idx], results[group_idx, :]))

    return return_results


class MaxLFQ:
    level: str
    threads: int
    minimum_subgroups: int

    def __init__(
        self, level: str = "protein", threads: int = 1, minimum_subgroups: int = 1
    ):
        self.level = level
        self.threads = threads
        self.minimum_subgroups = minimum_subgroups

        numba.set_num_threads(threads)

    def quantify(self, quant_matrix: QuantMatrix) -> pd.DataFrame:
        group_ids = numba.typed.List()
        key: str = ""

        if self.level == "protein":
            group_ids = numba.typed.List(quant_matrix.proteins)
            key = "Protein"

        elif self.level == "precursor":
            group_ids = numba.typed.List(quant_matrix.precursors)
            key = "PrecursorId"

        elif self.level == "peptide":
            group_ids = numba.typed.List(quant_matrix.peptides)
            key = "PeptideSequence"

        elif self.level in quant_matrix.quantitative_data.obs.columns:
            group_ids = numba.typed.List(
                quant_matrix.quantitative_data.obs[self.level].unique()
            )
            key = self.level
        else:
            raise ValueError("The level is not valid.")

        groupings = numba.typed.List()

        for group_id in group_ids:
            groupings.append(
                quant_matrix.quantitative_data[
                    quant_matrix.quantitative_data.obs[key] == group_id
                ].X.copy()
            )

        quantified_groups = quantify_groups(
            groupings, group_ids, self.minimum_subgroups
        )

        if self.level == "protein":
            results = self._format_result_df(
                quantified_groups, quant_matrix.sample_annotations["sample"].values
            )

        elif self.level == "precursor":
            results = self._format_result_df(
                quantified_groups, quant_matrix.sample_annotations["sample"].values
            )

            results = results.set_index(key).join(
                quant_matrix.quantitative_data.obs.set_index(key)
            )

        elif self.level == "peptide":
            results = self._format_result_df(
                quantified_groups, quant_matrix.sample_annotations["sample"].values
            )

            results = (
                results.set_index(key)
                .merge(
                    quant_matrix.quantitative_data.obs[
                        ~quant_matrix.quantitative_data.obs.duplicated(
                            ["PeptideSequence", "Protein"],
                        )
                    ].set_index(key),
                    how="left",
                    left_index=True,
                    right_index=True,
                )
                .reset_index()
            )
        else:
            results = self._format_result_df(
                quantified_groups, quant_matrix.sample_annotations["sample"].values
            )

        return results

    def _format_result_df(
        self,
        quantified_groups: List[Tuple[str, Union[np.ndarray, np.ndarray], bool]],
        samples: np.ndarray,
    ) -> pd.DataFrame:
        rows = []

        if self.level == "protein":
            label = "Protein"

        elif self.level == "precursor":
            label = "PrecursorId"

        elif self.level == "peptide":
            label = "PeptideSequence"

        else:
            label = self.level

        for result in quantified_groups:
            result_id = result[0]
            result_data = result[1]

            row = {
                label: result_id,
            }

            for idx, sample in enumerate(samples):
                row[sample] = result_data[idx]

            rows.append(row)

        return pd.DataFrame(rows)
