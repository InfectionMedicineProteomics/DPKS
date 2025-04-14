from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf  
import pandas as pd 

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any



class DifferentialTest:
    method: str
    min_samples_per_group: int
    level: str
    group_a: int
    group_b: int
    multiple_testing_correction_method: str
    covariates: Optional[List[str]]

    def __init__(
        self,
        method: str,
        comparisons: list,
        min_samples_per_group: int = 2,
        level: str = "precursor",
        multiple_testing_correction_method: str = "fdr_tsbh",
        covariates: Optional[List[str]] = None,
    ):
        self.method = method
        self.comparisons = comparisons
        self.min_samples_per_group = min_samples_per_group
        self.multiple_testing_correction_method = multiple_testing_correction_method
        self.covariates = covariates if covariates else []

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

        # Replace zeroes with NaN to avoid messing up stats
        quant_matrix.quantitative_data.X[
            quant_matrix.quantitative_data.X == 0.0
        ] = np.nan

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

            for identifier in identifiers:
                quant_data = quant_matrix.quantitative_data[
                    quant_matrix.row_annotations[self.level] == identifier, :
                ].copy()

                indices.append(quant_data.obs.index.to_numpy()[0])

                # Gather sample sets
                group_a_samples = quant_matrix.get_samples(group=group_a)
                if self.method == "ttest_paired":
                    group_b_samples = quant_matrix.get_pairs(samples=group_a_samples)
                else:
                    group_b_samples = quant_matrix.get_samples(group=group_b)

                group_a_data = quant_data[:, group_a_samples].X.copy()
                group_b_data = quant_data[:, group_b_samples].X.copy()

                # Count non-NaN
                group_a_nan = len(group_a_data[~np.isnan(group_a_data)])
                group_b_nan = len(group_b_data[~np.isnan(group_b_data)])

                group_a_rep_counts.append(group_a_nan)
                group_b_rep_counts.append(group_b_nan)

                # If either group doesn't meet min rep count, store NaN
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
                    continue

                # Otherwise, we drop NaNs for the actual test
                group_a_data = group_a_data[~np.isnan(group_a_data)]
                group_b_data = group_b_data[~np.isnan(group_b_data)]

                group_a_mean = np.mean(group_a_data)
                group_b_mean = np.mean(group_b_data)
                group_a_stdev = np.std(group_a_data)
                group_b_stdev = np.std(group_b_data)
                log_fold_change = group_a_mean - group_b_mean

                group_a_means.append(group_a_mean)
                group_b_means.append(group_b_mean)
                group_a_stdevs.append(group_a_stdev)
                group_b_stdevs.append(group_b_stdev)
                log_fold_changes.append(log_fold_change)

                expression_data = np.concatenate((group_a_data, group_b_data), axis=0)
                labels = np.array([group_a for _ in range(len(group_a_data))] +
                                  [group_b for _ in range(len(group_b_data))])

                if self.method == "ttest":
                    test_results = stats.ttest_ind(group_a_data, group_b_data)

                elif self.method == "ttest_paired":
                    test_results = stats.ttest_rel(group_a_data, group_b_data)

                elif self.method == "anova":
                    test_results = stats.f_oneway(group_a_data, group_b_data)

                elif self.method == "linregress":
                    if not self.covariates:
                        group_indicator = (labels == group_a).astype(int)
                        X = sm.add_constant(group_indicator)
                        model = sm.OLS(expression_data, X).fit() # switched to sm.OLS for consistency with covariates. Same as linregress
                        test_results = type("TestResults", (), {
                            "pvalue": model.pvalues["group_indicator"]
                        })
                    else:
                        group_indicator = (labels == group_a).astype(int)
                        all_samples = group_a_samples + group_b_samples

                        df = pd.DataFrame({
                            "expr": expression_data,
                            "group": group_indicator
                        })
                        
                        # Add covariates
                        cat_covariates = []
                        num_covariates = []
                        for covariate in self.covariates:
                            values = []
                            for sample in all_samples:
                                val = quant_matrix.sample_annotations.loc[sample, covariate]
                                values.append(val)
                            df[covariate] = values

                            if isinstance(values[0], str): # guess from first row
                                cat_covariates.append(covariate)
                            else:
                                num_covariates.append(covariate)

                        # Build the formula
                        # statsmodels uses R-like formulas
                        formula_terms = ["group"]
                        formula_terms += [f"C({covar})" for covar in cat_covariates]
                        formula_terms += num_covariates
                        formula = "expr ~ " + " + ".join(formula_terms)

                        model = smf.ols(formula, data=df).fit()
                        group_pval = model.pvalues["group"]

                        test_results = type("TestResults", (), {"pvalue": group_pval})

                p_values.append(test_results.pvalue)

            # Some columns for “-logp” or combined score
            log_p_values = [-np.log(p) if p is not None and p > 0 else np.nan
                            for p in p_values]
            max_log_p_value = np.nanmax(log_p_values)
            max_log_fold_change = np.nanmax([abs(fc) for fc in log_fold_changes])
            de_scores = [
                np.sqrt((p / max_log_p_value) ** 2 + (fc / max_log_fold_change) ** 2)
                if not np.isnan(p) and max_log_p_value != 0 else np.nan
                for p, fc in zip(log_p_values, log_fold_changes)
            ]

            # Write results to row_annotations
            quant_matrix.row_annotations[f"DEScore{group_a}-{group_b}"] = de_scores
            quant_matrix.row_annotations[f"Group{group_a}Mean"] = group_a_means
            quant_matrix.row_annotations[f"Group{group_b}Mean"] = group_b_means
            quant_matrix.row_annotations[f"Group{group_a}Stdev"] = group_a_stdevs
            quant_matrix.row_annotations[f"Group{group_b}Stdev"] = group_b_stdevs
            quant_matrix.row_annotations[f"Log2FoldChange{group_a}-{group_b}"] = log_fold_changes
            quant_matrix.row_annotations[f"PValue{group_a}-{group_b}"] = p_values
            quant_matrix.row_annotations[f"Group{group_a}RepCounts"] = group_a_rep_counts
            quant_matrix.row_annotations[f"Group{group_b}RepCounts"] = group_b_rep_counts

            # Correct for multiple testing
            quant_matrix.quantitative_data.obs.sort_values(
                f"PValue{group_a}-{group_b}", inplace=True
            )

            valid_pvals = quant_matrix.quantitative_data.obs[
                ~np.isnan(quant_matrix.quantitative_data.obs[f"PValue{group_a}-{group_b}"])
            ][f"PValue{group_a}-{group_b}"]

            correction_results = multipletests(
                valid_pvals,
                method=self.multiple_testing_correction_method,
                is_sorted=False,
            )

            corrected = np.full(len(quant_matrix.quantitative_data.obs), np.nan)
            corrected[: len(correction_results[1])] = correction_results[1]

            quant_matrix.quantitative_data.obs[f"CorrectedPValue{group_a}-{group_b}"] = corrected
            quant_matrix.quantitative_data.obs[f"-Log10CorrectedPValue{group_a}-{group_b}"] = -np.log10(
                corrected
            )

            # Restore original sort order
            quant_matrix.quantitative_data.obs.index = (
                quant_matrix.quantitative_data.obs.index.map(int)
            )
            quant_matrix.quantitative_data.obs.sort_index(inplace=True)
            quant_matrix.quantitative_data.obs.index = (
                quant_matrix.quantitative_data.obs.index.map(str)
            )

        return quant_matrix
