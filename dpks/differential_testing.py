from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats  # type: ignore

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any

from statsmodels.stats.multitest import multipletests  # type: ignore


class DifferentialTest:
    method: str
    min_samples_per_group: int
    level: str
    group_a: int
    group_b: int
    multiple_testing_correction_method: str

    def __init__(
        self,
        method: str,
        comparisons: list,
        min_samples_per_group: int = 2,
        level: str = "precursor",
        multiple_testing_correction_method: str = "fdr_tsbh",
    ):
        self.method = method
        self.comparisons = comparisons
        self.min_samples_per_group = min_samples_per_group
        self.multiple_testing_correction_method = multiple_testing_correction_method
        if level == "precursor":
            self.level = "PrecursorId"

        elif level == "protein":
            self.level = "Protein"

        elif level == "peptide":
            self.level = "PeptideSequence"
        else:
            self.level = level

    def test(self, quant_matrix: QuantMatrix) -> QuantMatrix:
        if self.level == "PrecursorId":
            identifiers = quant_matrix.precursors

        elif self.level == "Protein":
            identifiers = quant_matrix.proteins

        elif self.level == "PeptideSequence":
            identifiers = quant_matrix.peptides
        else:
            identifiers = quant_matrix.quantitative_data.obs[self.level]

        if isinstance(self.comparisons, tuple):
            self.comparisons = [self.comparisons]

        for comparison in self.comparisons:
            group_a, group_b = comparison
            group_a_means = []
            group_b_means = []
            group_a_stdevs = []
            group_b_stdevs = []
            log_fold_changes = []
            p_values = []
            group_a_rep_counts = []
            group_b_rep_counts = []
            indices = []

            quant_matrix.quantitative_data.X[
                quant_matrix.quantitative_data.X == 0.0
            ] = np.nan

            for identifier in identifiers:
                quant_data = quant_matrix.quantitative_data[
                    quant_matrix.row_annotations[self.level] == identifier, :
                ].copy()

                indices.append(quant_data.obs.index.to_numpy()[0])

                group_a_samples = quant_matrix.get_samples(group=group_a)
                if self.method == "ttest_paired":
                    group_b_samples = quant_matrix.get_pairs(samples=group_a_samples)
                else:
                    group_b_samples = quant_matrix.get_samples(group=group_b)

                group_a_data = quant_data[:, group_a_samples].X.copy()
                group_b_data = quant_data[:, group_b_samples].X.copy()

                group_a_nan = len(group_a_data[~np.isnan(group_a_data)])
                group_b_nan = len(group_b_data[~np.isnan(group_b_data)])

                group_a_rep_counts.append(group_a_nan)
                group_b_rep_counts.append(group_b_nan)

                if (group_a_nan < self.min_samples_per_group) or (
                    group_b_nan < self.min_samples_per_group
                ):
                    if group_a_nan < self.min_samples_per_group:
                        group_a_means.append(np.nan)
                        group_a_stdevs.append(np.nan)

                    else:
                        group_a_means.append(np.mean(group_a_data))
                        group_a_stdevs.append(np.std(group_a_data))

                    if group_b_nan < self.min_samples_per_group:
                        group_b_means.append(np.nan)
                        group_b_stdevs.append(np.nan)

                    else:
                        group_b_means.append(np.mean(group_b_data))
                        group_b_stdevs.append(np.std(group_b_data))

                    log_fold_changes.append(np.nan)
                    p_values.append(np.nan)

                else:
                    group_a_data = group_a_data[~np.isnan(group_a_data)]
                    group_b_data = group_b_data[~np.isnan(group_b_data)]

                    group_a_mean = np.mean(group_a_data)
                    group_b_mean = np.mean(group_b_data)
                    group_a_stdev = np.std(group_a_data)
                    group_b_stdev = np.std(group_b_data)

                    log_fold_change = group_a_mean - group_b_mean

                    group_a_labels = np.array([1.0 for _ in range(len(group_a_data))])
                    group_b_labels = np.array([2.0 for _ in range(len(group_b_data))])

                    labels = np.concatenate([group_a_labels, group_b_labels])

                    expression_data = np.concatenate(
                        [group_a_data, group_b_data], axis=0
                    )

                    if self.method == "ttest":
                        test_results = stats.ttest_ind(group_a_data, group_b_data)

                    elif self.method == "linregress":
                        test_results = stats.linregress(x=expression_data, y=labels)

                    elif self.method == "anova":
                        test_results = stats.f_oneway(group_a_data, group_b_data)

                    elif self.method == "ttest_paired":
                        test_results = stats.ttest_rel(group_a_data, group_b_data)

                    group_a_means.append(group_a_mean)
                    group_b_means.append(group_b_mean)
                    group_a_stdevs.append(group_a_stdev)
                    group_b_stdevs.append(group_b_stdev)
                    log_fold_changes.append(log_fold_change)
                    p_values.append(test_results.pvalue)

            log_p_values = [-np.log(p) for p in p_values]
            max_log_p_value = np.nanmax(log_p_values)
            max_log_fold_change = np.nanmax([abs(fc) for fc in log_fold_changes])
            de_scores = [
                np.sqrt((p / max_log_p_value) ** 2 + (fc / max_log_fold_change) ** 2)
                for p, fc in zip(log_p_values, log_fold_changes)
            ]
            quant_matrix.row_annotations[f"DEScore{group_a}-{group_b}"] = de_scores
            quant_matrix.row_annotations[f"Group{group_a}Mean"] = group_a_means
            quant_matrix.row_annotations[f"Group{group_b}Mean"] = group_b_means
            quant_matrix.row_annotations[f"Group{group_a}Stdev"] = group_a_stdevs
            quant_matrix.row_annotations[f"Group{group_b}Stdev"] = group_b_stdevs
            quant_matrix.row_annotations[f"Log2FoldChange{group_a}-{group_b}"] = (
                log_fold_changes
            )
            quant_matrix.row_annotations[f"PValue{group_a}-{group_b}"] = p_values
            quant_matrix.row_annotations[f"Group{group_a}RepCounts"] = (
                group_a_rep_counts
            )
            quant_matrix.row_annotations[f"Group{group_b}RepCounts"] = (
                group_b_rep_counts
            )

            quant_matrix.quantitative_data.obs.sort_values(
                f"PValue{group_a}-{group_b}", inplace=True
            )

            correction_results = multipletests(
                quant_matrix.quantitative_data.obs[
                    ~np.isnan(
                        quant_matrix.quantitative_data.obs[f"PValue{group_a}-{group_b}"]
                    )
                ][f"PValue{group_a}-{group_b}"],
                method=self.multiple_testing_correction_method,
                is_sorted=False,
            )

            corrected_results = np.empty(
                (len(quant_matrix.quantitative_data.obs),), dtype=np.float64
            )
            corrected_results[:] = np.nan

            corrected_results[: len(correction_results[1])] = correction_results[1]

            quant_matrix.quantitative_data.obs[
                f"CorrectedPValue{group_a}-{group_b}"
            ] = corrected_results

            quant_matrix.quantitative_data.obs[
                f"-Log10CorrectedPValue{group_a}-{group_b}"
            ] = -np.log10(corrected_results)

            quant_matrix.quantitative_data.obs.index = (
                quant_matrix.quantitative_data.obs.index.map(int)
            )
            quant_matrix.quantitative_data.obs.sort_index(inplace=True)
            quant_matrix.quantitative_data.obs.index = (
                quant_matrix.quantitative_data.obs.index.map(str)
            )

        return quant_matrix
