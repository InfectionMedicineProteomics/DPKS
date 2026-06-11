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

def _correct_pvalues(pvales, method="fdr_tsbh"):

    _, corrected_pvals, _, _ = multipletests(pvales, method=method)

    return corrected_pvals

class FastOLS:
    residuals: np.ndarray
    fitted_values: np.ndarray
    pvalues: np.ndarray
    params: np.ndarray
    fit_intercept: bool
    multiple_testing_correction_method: str

    def __init__(self, fit_intercept: bool = True, multiple_testing_correction_method: str = "fdr_tsbh"):
        self.residuals = None
        self.fitted_values = None
        self.pvalues = None
        self.params = None
        self.fit_intercept = fit_intercept
        self.multiple_testing_correction_method = multiple_testing_correction_method

    @property
    def corrected_pvalues(self):

        return np.apply_along_axis(
            _correct_pvalues, axis=1, arr=self.pvalues, method=self.multiple_testing_correction_method
        )

    def fit(self, X, Z):

        if self.fit_intercept:
            Z = np.concatenate(
                [np.ones(X.shape[0]).reshape(-1, 1), Z],
                axis=1
            )

        q = Z.shape[1]
        n = X.shape[0]
        p = X.shape[1]

        params = np.full((q, p), np.nan)
        pvalues = np.full((q, p), np.nan)
        fitted_values = np.full((n, p), np.nan)
        residuals = np.full((n, p), np.nan)

        obs_matrix = ~np.isnan(X)
        pattern_ids = np.packbits(obs_matrix, axis=0)

        _, inverse, counts = np.unique(
            pattern_ids.T, axis=0, return_inverse=True, return_counts=True
        )

        for grp_idx in np.unique(inverse):

            cols_ = np.where(inverse == grp_idx)[0]

            X_subset = X[:, cols_]
            X_mask = ~np.isnan(X_subset[:, 0], )

            Z_subset = Z[X_mask]
            X_subset = X_subset[X_mask]

            n_subset = X_subset.shape[0]

            Q, R = np.linalg.qr(Z_subset)
            QtX = Q.T @ X_subset
            B_hat = np.linalg.solve(R, QtX)
            X_fitted = Z_subset @ B_hat

            _residuals = X_subset - X_fitted
            sigma2 = np.sum(_residuals ** 2, axis=0) / (n_subset - q)

            r_inv = np.linalg.inv(R)
            diag_inv = np.sum(r_inv ** 2, axis=1)
            standard_error = np.sqrt(np.outer(diag_inv, sigma2))
            t_statistics = B_hat / standard_error

            degrees_of_freedom = n_subset - q
            _pvalues = 2 * stats.t.sf(np.abs(t_statistics), df=degrees_of_freedom)

            params[:, cols_] = B_hat
            pvalues[:, cols_] = _pvalues
            fitted_values[np.ix_(X_mask, cols_)] = X_fitted
            residuals[np.ix_(X_mask, cols_)] = _residuals

        self.params = params
        self.pvalues = pvalues
        self.fitted_values = fitted_values
        self.residuals = residuals


class DifferentialTest:
    method: str
    min_samples_per_group: int
    level: str
    group_a: int
    group_b: int
    multiple_testing_correction_method: str
    covariates: Optional[List[str]]
    log2_transformed: bool

    def __init__(
        self,
        method: str,
        comparison: tuple[Any, Any],
        min_samples_per_group: int = 2,
        level: str = "precursor",
        multiple_testing_correction_method: str = "fdr_tsbh",
        covariates: Optional[List[str]] = None,
        log2_transformed: bool = True,
    ):
        self.method = method
        self.comparison = comparison
        self.min_samples_per_group = min_samples_per_group
        self.multiple_testing_correction_method = multiple_testing_correction_method
        self.covariates = covariates if covariates else []
        self.log2_transformed = log2_transformed

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

        group_a, group_b = self.comparison
        group_a_means = []
        group_b_means = []
        group_a_stdevs = []
        group_b_stdevs = []
        log_fold_changes = []
        p_values = []
        group_a_rep_counts = []
        group_b_rep_counts = []
        indices = []

        if self.method == "fast_ols":
            ols = FastOLS(
                fit_intercept=True,
            )

            #This block is to ensure that only the correct samples are taken
            group_a_samples = quant_matrix.get_samples(group=group_a)
            group_b_samples = quant_matrix.get_samples(group=group_b)
            samples = group_a_samples + group_b_samples
            quant_data = quant_matrix.quantitative_data[
                :, quant_matrix.quantitative_data.var['sample'].isin(samples)
            ].copy()

            X = quant_data.X.T
            design_matrix = quant_data.var[['group'] + self.covariates].copy()

            original_groups = design_matrix['group'].to_numpy()
            design_matrix['group'] = (design_matrix['group'] == group_a).astype(int)

            ols.fit(X, design_matrix)

            group_a_idx = np.argwhere(original_groups == group_a)
            group_b_idx = np.argwhere(original_groups == group_b)

            group_a_means = np.nanmean(X[group_a_idx], axis=0).ravel()
            group_b_means = np.nanmean(X[group_b_idx], axis=0).ravel()

            group_a_stdevs = np.nanstd(X[group_a_idx], axis=0).ravel()
            group_b_stdevs = np.nanstd(X[group_b_idx], axis=0).ravel()

            log_fold_changes = ols.params[1, :]
            p_values = ols.pvalues[1, :]

            group_a_rep_counts = np.sum(~np.isnan(X[group_a_idx]), axis=0).ravel()
            group_b_rep_counts = np.sum(~np.isnan(X[group_b_idx]), axis=0).ravel()

        else:

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

                # Sets 0 to np.nan so that things can be calculated nicely
                group_a_data = np.where(group_a_data == 0, np.nan, group_a_data)
                group_b_data = np.where(group_b_data == 0, np.nan, group_b_data)

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

                if self.log2_transformed:

                    log_fold_change = group_a_mean - group_b_mean

                else:

                    log_fold_change = group_a_mean / group_b_mean

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
                        X = pd.DataFrame({"const": np.ones(len(group_indicator)),
                                          "group_indicator": group_indicator})
                        model = sm.OLS(expression_data,
                                       X).fit()  # switched to sm.OLS for consistency with covariates. Same as linregress
                        test_results = type("TestResults", (), {
                            "pvalue": model.pvalues["group_indicator"],
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

                            if isinstance(values[0], str):  # guess from first row
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
