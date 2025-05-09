"""quant_matrix module

instanciate a quant matrix:

>>> from dpks.quant_matrix import QuantMatrix
>>> quant_matrix = QuantMatrix( quantification_file="tests/input_files/minimal_matrix.tsv", design_matrix_file="tests/input_files/minimal_design_matrix.tsv")

"""

from __future__ import annotations

from typing import Union, List, Any, Optional

import anndata as ad
import gseapy as gp
import matplotlib
import numpy as np
import pandas as pd  # type: ignore
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder

from dpks.annotation import get_protein_labels, get_genes_from_proteins
from dpks.clustering import FeatureClustering
from dpks.correction import CorrectionMethod, MeanCorrection
from dpks.differential_testing import DifferentialTest
from dpks.fdr import DecoyCounter
from dpks.fdr import DecoyFeatures, MeanDecoyFeatures, ShuffleDecoyFeatures
from dpks.imputer import (
    ImputerMethod,
    NeighborhoodImputer,
    UniformRangeImputer,
    UniformPercentileImputer,
    ConstantImputer,
)
from dpks.interpretation import (
    BootstrapInterpreter,
    Classifier,
    encode_labels,
    format_data,
    TrainResult,
)
from dpks.normalization import (
    TicNormalization,
    MedianNormalization,
    MeanNormalization,
    Log2Normalization,
    NormalizationMethod,
    RTSlidingWindowNormalization,
)
from dpks.param_search import GeneticAlgorithmSearch, RandomizedSearch, ParamSearchResult  # type: ignore
from dpks.parsers import parse_diann
from dpks.plot import ImportancePlot, RFEPCA
from dpks.quantification import TopN, MaxLFQ
from dpks.scaling import (
    ScalingMethod,
    ZScoreScaling,
    MinMaxScaling,
    AbsMaxScaling,
)


class QuantMatrix:
    """Class for working with quantitative matrices."""

    quantification_file_path: Union[str, pd.DataFrame]
    design_matrix_file: Union[str, pd.DataFrame]
    num_rows: int
    num_samples: int
    quantitative_data: ad.AnnData
    explain_results: Optional[list[tuple[Any, BootstrapInterpreter]]]

    def __init__(
        self,
        quantification_file: Union[str, pd.DataFrame],
        design_matrix_file: Union[str, pd.DataFrame],
        annotation_fasta_file: str = None,
        quant_type: str = "gps",
        diann_qvalue: float = 0.01,
    ) -> None:
        """Initialize the QuantMatrix instance.

        Args:
            quantification_file (Union[str, pd.DataFrame]): Path to the quantification file or DataFrame.
            design_matrix_file (Union[str, pd.DataFrame]): Path to the design matrix file or DataFrame.
            annotation_fasta_file (str, optional): Path to the annotation FASTA file. Defaults to None.
            quant_type (str, optional): Type of quantification. Defaults to "gps".
            diann_qvalue (float, optional): DIANN q-value. Defaults to 0.01.

        Examples:
            >>> quant_matrix = QuantMatrix("quantification.tsv", "design_matrix.csv", annotation_fasta_file="annotation.fasta")
        """

        self.annotated = False
        self.explain_results = None
        if isinstance(design_matrix_file, str):
            design_matrix_file = pd.read_csv(design_matrix_file, sep="\t")

            design_matrix_file.columns = map(str.lower, design_matrix_file.columns)

        if isinstance(quantification_file, str):
            if quant_type == "gps":
                quantification_file = pd.read_csv(quantification_file, sep="\t")

            elif quant_type == "diann":
                quantification_file = parse_diann(quantification_file, diann_qvalue)

        else:
            if quant_type == "diann":
                quantification_file = parse_diann(quantification_file, diann_qvalue)

        self.num_samples = len(design_matrix_file)
        self.num_rows = len(quantification_file)

        rt_column = ""

        if "RT" in quantification_file:
            rt_column = "RT"

        elif "RetentionTime" in quantification_file:
            rt_column = "RetentionTime"

        if rt_column:
            quantification_file = quantification_file.sort_values(rt_column)

        quantification_file = quantification_file.reset_index(drop=True)

        quantitative_data = (
            quantification_file[list(design_matrix_file["sample"])]
            .copy()
            .set_index(np.arange(self.num_rows, dtype=int).astype(str))
        )

        row_obs = quantification_file.drop(
            list(design_matrix_file["sample"]), axis=1
        ).set_index(np.arange(self.num_rows, dtype=int).astype(str))

        if annotation_fasta_file is not None:
            row_obs["ProteinLabel"] = get_protein_labels(
                row_obs["Protein"], annotation_fasta_file
            )

        self.quantitative_data = ad.AnnData(
            quantitative_data,
            obs=row_obs,
            var=design_matrix_file.copy().set_index(design_matrix_file["sample"]),
        )

    @property
    def proteins(self) -> List[str]:
        return list(self.quantitative_data.obs["Protein"].unique())

    @property
    def protein_labels(self) -> List[str]:
        return self.row_annotations["ProteinLabel"].to_list()

    @property
    def sample_groups(self) -> List[str]:
        return self.sample_annotations["group"].to_list()

    @property
    def peptides(self) -> List[str]:
        return list(self.quantitative_data.obs["PeptideSequence"].unique())

    @property
    def precursors(self) -> List[str]:
        self.row_annotations["PrecursorId"] = (
            self.row_annotations["PeptideSequence"]
            + "_"
            + self.row_annotations["Charge"].astype(str)
        )

        return list(self.row_annotations["PrecursorId"].unique())

    @property
    def sample_annotations(self) -> pd.DataFrame:
        return self.quantitative_data.var

    @property
    def row_annotations(self) -> pd.DataFrame:
        return self.quantitative_data.obs

    @row_annotations.setter
    def row_annotations(self, value: pd.DataFrame) -> None:
        self.quantitative_data.obs = value

    def get_samples(self, group=None) -> List[str]:
        if group:
            return list(
                self.sample_annotations[self.sample_annotations["group"] == group][
                    "sample"
                ]
            )
        else:
            return self.sample_annotations["sample"]

    def get_pairs(self, samples: list) -> List[str]:
        sorted_samples = (
            self.sample_annotations[self.sample_annotations["sample"].isin(samples)]
            .set_index("sample")
            .loc[samples]
        )

        return list(sorted_samples["pair"])

    def get_batches(self) -> np.ndarray:
        return self.sample_annotations["batch"].values

    def filter(
        self,
        peptide_q_value: float = 0.01,
        protein_q_value: float = 0.01,
        remove_decoys: bool = True,
        remove_contaminants: bool = True,
        remove_non_proteotypic: bool = True,
        remove_zero_rows: bool = True,
        remove_n_zero_rows: bool = False,
        max_n_zeros: int = None,
    ) -> QuantMatrix:
        """Filter the QuantMatrix.

        Args:
            peptide_q_value (float, optional): Peptide q-value threshold. Defaults to 0.01.
            protein_q_value (float, optional): Protein q-value threshold. Defaults to 0.01.
            remove_decoys (bool, optional): Whether to remove decoy entries. Defaults to True.
            remove_contaminants (bool, optional): Whether to remove contaminant entries. Defaults to True.
            remove_non_proteotypic (bool, optional): Whether to remove non-proteotypic entries. Defaults to True.

        Returns:
            QuantMatrix: Filtered QuantMatrix object.

        Examples:
            >>> print(quant_matrix.to_df().shape)
            (16679, 26)
            >>> print(quant_matrix.filter(peptide_q_value=0.001).to_df().shape)
            (15355, 26)

        """

        filtered_data = self.quantitative_data

        if "PeptideQValue" in self.quantitative_data.obs:
            filtered_data = self.quantitative_data[
                (self.quantitative_data.obs["PeptideQValue"] <= peptide_q_value)
            ].copy()

        if "ProteinQValue" in self.quantitative_data.obs:
            filtered_data = self.quantitative_data[
                (self.quantitative_data.obs["ProteinQValue"] <= protein_q_value)
            ].copy()

        if remove_decoys:
            if "Decoy" in filtered_data.obs:
                filtered_data = filtered_data[filtered_data.obs["Decoy"] == 0].copy()

        if remove_contaminants:
            filtered_data = filtered_data[
                ~filtered_data.obs["Protein"].str.contains("contam")
            ].copy()

            filtered_data = filtered_data[
                ~filtered_data.obs["Protein"].str.contains("cont_crap")
            ].copy()

        if remove_non_proteotypic:
            filtered_data = filtered_data[
                ~filtered_data.obs["Protein"].str.contains(";")
            ].copy()

        if remove_zero_rows:
            X_nan_to_num = np.nan_to_num(filtered_data.X, nan=0)
            non_zero_rows_mask = ~np.all(X_nan_to_num == 0, axis=1)
            filtered_data = filtered_data[non_zero_rows_mask].copy()

        if remove_n_zero_rows:
            if max_n_zeros == None:
                raise ValueError(
                    "If remove proteins with more than n zeros, must pass max_n_zeros."
                )
            X_nan_to_num = np.nan_to_num(filtered_data.X, nan=0)
            zero_counts = np.sum(X_nan_to_num == 0, axis=1)
            rows_to_keep = zero_counts <= max_n_zeros
            filtered_data = filtered_data[rows_to_keep].copy()

        self.num_rows = len(filtered_data)

        quantitative_data = (
            filtered_data.to_df()[list(filtered_data.var["sample"])]
            .copy()
            .set_index(np.arange(self.num_rows, dtype=int).astype(str))
        )

        row_obs = filtered_data.obs.set_index(
            np.arange(self.num_rows, dtype=int).astype(str)
        )

        self.quantitative_data = ad.AnnData(
            quantitative_data, obs=row_obs, var=filtered_data.var
        )

        return self

    def scale(
        self,
        method: str,
    ) -> QuantMatrix:
        """Scale the QuantMatrix data at the feature level (i.e Precursor or Protein).

        Args:
            method (str): Scaling method. Options are 'zscore', 'minmax', or 'absmax'.

        Returns:
            QuantMatrix: Scaled QuantMatrix object.

        Raises:
            ValueError: If the provided scaling method is not supported.

        """
        base_method: ScalingMethod = ScalingMethod()

        if method == "zscore":
            base_method = ZScoreScaling()

        elif method == "minmax":
            base_method = MinMaxScaling()

        elif method == "absmax":
            base_method = AbsMaxScaling()

        else:
            raise ValueError(f"Unsupported scaling method: {method}")

        self.quantitative_data.X = base_method.fit_transform(self.quantitative_data.X)

        return self

    def normalize(
        self,
        method: str,
        log_transform: bool = True,
        use_rt_sliding_window_filter: bool = False,
        **kwargs: Union[int, bool, str],
    ) -> QuantMatrix:
        """Normalize the QuantMatrix data.

        Args:
            method (str): Normalization method. Options are 'tic', 'median', or 'mean'.
            log_transform (bool, optional): Whether to log-transform the data. Defaults to True.
            use_rt_sliding_window_filter (bool, optional): Whether to use a sliding window filter. Defaults to False. Can only use if a RetentionTime column was loaded in the QuantMatrix
            **kwargs: Additional keyword arguments depending on the chosen method.

        Returns:
            QuantMatrix: Normalized QuantMatrix object.

        Raises:
            ValueError: If the provided normalization method is not supported.

        Examples:
            >>> quant_matrix.normalize(method="mean")

        """

        base_method: NormalizationMethod = NormalizationMethod()

        if method == "tic":
            base_method = TicNormalization()

        elif method == "median":
            base_method = MedianNormalization()

        elif method == "mean":
            base_method = MeanNormalization()
        elif method == "log2":
            base_method = Log2Normalization()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        if use_rt_sliding_window_filter:
            minimum_data_points = int(kwargs.get("minimum_data_points", 100))
            stride = int(kwargs.get("stride", 1))
            use_overlapping_windows = bool(kwargs.get("use_overlapping_windows", True))
            rt_unit = str(kwargs.get("rt_unit", "minute"))

            rt_window_normalization = RTSlidingWindowNormalization(
                base_method=base_method,
                minimum_data_points=minimum_data_points,
                stride=stride,
                use_overlapping_windows=use_overlapping_windows,
                rt_unit=rt_unit,
            )

            self.quantitative_data.X = rt_window_normalization.fit_transform(self)

        else:
            self.quantitative_data.X = base_method.fit_transform(
                self.quantitative_data.X
            )

        if log_transform and not (method == "log2"):
            self.quantitative_data.X = Log2Normalization().fit_transform(
                self.quantitative_data.X
            )

        return self

    def correct(self, method: str = "mean", reference_batch=None):
        base_method: CorrectionMethod = CorrectionMethod()
        batches = self.get_batches()

        if method == "mean":
            if reference_batch not in batches:
                raise ValueError("The reference batch is not one of the batches.")

            base_method = MeanCorrection(reference_batch=reference_batch)

        self.quantitative_data.X = base_method.fit_transform(
            self.quantitative_data.X, batches
        )

        return self

    def quantify(
        self,
        method: str,
        **kwargs: Union[int, str],
    ) -> QuantMatrix:
        """Calculate protein quantities.

        Args:
            method (str): Quantification method. Options are 'top_n' or 'maxlfq'.
            **kwargs: Additional keyword arguments depending on the chosen method.

        Returns:
            QuantMatrix: Quantified protein matrix.

        Raises:
            ValueError: If the provided quantification method is not supported.

        Examples:
            >>> quant_matrix.quantify(method="top_n", top_n=1)

        """

        if method == "top_n":
            level = str(kwargs.get("level", "protein"))
            top_n = int(kwargs.get("top_n", 1))
            summarization_method = str(kwargs.get("summarization_method", "sum"))

            quantifications = TopN(
                top_n=top_n, level=level, summarization_method=summarization_method
            ).quantify(self)

            design_matrix = self.quantitative_data.var

            protein_quantifications = QuantMatrix(
                quantifications, design_matrix_file=design_matrix
            )

        elif method == "maxlfq":
            level = str(kwargs.get("level", "protein"))
            threads = int(kwargs.get("threads", 1))
            minimum_subgroups = int(kwargs.get("minimum_subgroups", 1))
            top_n = int(kwargs.get("top_n", 0))

            quantifications = MaxLFQ(
                level=level,
                threads=threads,
                minimum_subgroups=minimum_subgroups,
                top_n=top_n,
            ).quantify(self)

            design_matrix = self.quantitative_data.var

            protein_quantifications = QuantMatrix(
                quantifications, design_matrix_file=design_matrix
            )

        else:
            raise ValueError(f"Unsupported quantification method: {method}")

        return protein_quantifications

    def impute(self, method: str, **kwargs: int) -> QuantMatrix:
        """Impute missing values in the quantitative data.

        Args:
            method (str): The imputation method to use. Options are "uniform_percentile" and "uniform_range"
            **kwargs (int): Additional keyword arguments specific to the imputation method.

        Returns:
            QuantMatrix: The QuantMatrix object with missing values imputed.

        Raises:
            ValueError: If an unsupported imputation method is provided.

        Examples:
            >>> quant_matrix.impute(method="uniform_percentile", percentile=0.1)

        """

        base_method: ImputerMethod = ImputerMethod()

        if method == "uniform_percentile":
            percentile = float(kwargs.get("percentile", 0.1))

            base_method = UniformPercentileImputer(percentile=percentile)

        elif method == "uniform_range":
            maxvalue = float(kwargs.get("maxvalue", 1))
            minvalue = float(kwargs.get("minvalue", 0))

            base_method = UniformRangeImputer(maxvalue=maxvalue, minvalue=minvalue)

        elif method == "constant":
            constant = float(kwargs.get("constant", 0))

            base_method = ConstantImputer(constant=constant)

        elif method == "neighborhood":
            n_neighbors = int(kwargs.get("n_neighbors", 5))
            weights = str(kwargs.get("weights", "distance"))

            base_method = NeighborhoodImputer(n_neighbors=n_neighbors, weights=weights)

        else:
            raise ValueError(f"Unsupported imputation method: {method}")

        self.quantitative_data.X = base_method.fit_transform(self.quantitative_data.X)

        return self

    def append(
        self, method: str = "mean", feature_column: str = "Protein"
    ) -> QuantMatrix:
        self.decoy_features = DecoyFeatures()

        if method == "mean":
            X, y = self.to_ml(feature_column=feature_column)

            n_samples = X.shape[0]
            n_features = X.shape[1]

            self.decoy_features = MeanDecoyFeatures(
                n_samples=n_samples, n_features=n_features, feature_names=X.columns
            )

            self.decoy_features.fit(X)

        elif method == "shuffle":
            X, y = self.to_ml(feature_column=feature_column)

            n_samples = X.shape[0]
            n_features = X.shape[1]

            self.decoy_features = ShuffleDecoyFeatures(
                n_samples=n_samples, n_features=n_features, feature_names=X.columns
            )

            self.decoy_features.fit(X)

        decoy_df = self.decoy_features.features.T

        id_columns = ["Protein", "ProteinLabel", "Gene"]

        id_df = self.row_annotations[
            [col for col in self.row_annotations.columns if col in id_columns]
        ].set_index(feature_column)

        used_id_columns = list(id_df.columns) + [feature_column]

        decoy_df = decoy_df.join(id_df).reset_index(names=[feature_column])

        for col in used_id_columns:
            decoy_df[col] = "decoy_" + decoy_df[col]

        target_df = self.to_df()

        combined_features = pd.concat([target_df, decoy_df], axis=0)

        combined_features["Decoy"] = np.where(
            combined_features[feature_column].str.contains("decoy"), 1, 0
        )

        return QuantMatrix(
            quantification_file=combined_features.copy(),
            design_matrix_file=self.quantitative_data.var.copy(),
        )

    def compare(
        self,
        method: str,
        comparisons: list,
        min_samples_per_group: int = 2,
        level: str = "protein",
        multiple_testing_correction_method: str = "fdr_tsbh",
        covariates: Optional[List[str]] = None,
    ) -> QuantMatrix:
        """Compare groups by differential testing.

        Args:
            method (str): Statistical comparison method. Options are 'ttest', 'linregress', 'anova', 'ttest_paired'.
            comparisons (list): List of tuples specifying the group comparisons.
            min_samples_per_group (int, optional): Minimum number of samples per group. Defaults to 2.
            level (str, optional): Level of comparison. Defaults to 'protein'.
            multiple_testing_correction_method (str, optional): Method for multiple testing correction. Defaults to 'fdr_tsbh'.
            covariates (Optional[List[str]], optional): List of column names in the design matrix to use as covariates. Defaults to None.

        Returns:
            QuantMatrix: Matrix containing the results of the differential testing.

        Raises:
            ValueError: If the provided statistical comparison method is not supported.

        Examples:
            >>> quantified_data = quantified_data.compare(
            >>>     method="linregress",
            >>>     min_samples_per_group=2,
            >>>     comparisons=[(2, 1), (3, 1)],
            >>>     covariates=["gender", "age"]
            >>> )
        """

        if not method in {"ttest", "linregress", "anova", "ttest_paired"}:
            raise ValueError(f"Unsupported statistical comparison method: {method}")

        # Check that all covariates are in the sample annotations
        if covariates:
            for covariate in covariates:
                if covariate not in self.sample_annotations.columns:
                    raise ValueError(f"Covariate '{covariate}' not found in sample annotations")

        differential_test = DifferentialTest(
            method,
            comparisons,
            min_samples_per_group,
            level,
            multiple_testing_correction_method,
            covariates,
        )

        compared_data = differential_test.test(self)

        self.row_annotations = compared_data.row_annotations.copy()

        return self

    def explain(
        self,
        clf,
        comparisons: list,
        n_iterations: int = 100,
        downsample_background: bool = True,
        feature_column: str = "Protein",
        fillna: bool = True,
        use_sample_weight: bool = True,
    ) -> QuantMatrix:
        """Explain group differences using explainable machine learning and feature importance.

        Args:
            clf: Classifier object used for prediction.
            comparisons (list): List of tuples specifying the group comparisons.
            n_iterations (int, optional): Number of iterations for bootstrapping. Defaults to 100.
            downsample_background (bool, optional): Whether to downsample the background. Defaults to True.
            feature_column (str, optional): Name of the feature column. Defaults to 'Protein'.

        Returns:
            QuantMatrix: Matrix containing the results of the explanation.

        Examples:
            >>> import xgboost
            >>>
            >>> clf = xgboost.XGBClassifier(
            >>>     max_depth=2,
            >>>     reg_lambda=2,
            >>>     objective="binary:logistic",
            >>>     seed=42
            >>> )
            >>>
            >>> quantified_data = quantified_data.explain(
            >>>     clf,
            >>>     comparisons=[(2, 1), (3, 1)],
            >>>     n_iterations=10,
            >>>     downsample_background=True
            >>> )

        """
        explain_results = []

        if isinstance(comparisons, tuple):
            comparisons = [comparisons]

        for comparison in comparisons:
            X, y = self.to_ml(feature_column=feature_column, comparison=comparison)

            scaler = StandardScaler()

            if fillna:
                X[:] = X[:].fillna(0.0)

            X[:] = scaler.fit_transform(X[:])

            interpreter = BootstrapInterpreter(
                feature_names=X.columns,
                n_iterations=n_iterations,
                downsample_background=downsample_background,
            )

            interpreter.fit(X.values, y.values.ravel(), clf)

            explain_results.append((comparison, interpreter))

            importances_df = interpreter.results_[
                [
                    "feature",
                    "mean_importance",
                    "mean_rank",
                    "median_importance",
                    "stdev_importance",
                    "median_rank",
                    "stdev_rank",
                ]
            ].set_index("feature")

            importances_df = importances_df.rename(
                columns={
                    "mean_importance": f"MeanImportance{comparison[0]}-{comparison[1]}",
                    "mean_rank": f"MeanRank{comparison[0]}-{comparison[1]}",
                    "median_importance": f"MedianImportance{comparison[0]}-{comparison[1]}",
                    "stdev_importance": f"StdevImportance{comparison[0]}-{comparison[1]}",
                    "median_rank": f"MedianRank{comparison[0]}-{comparison[1]}",
                    "stdev_rank": f"StdevRank{comparison[0]}-{comparison[1]}",
                }
            )

            self.row_annotations = self.row_annotations.join(
                importances_df, on=feature_column
            )

        self.explain_results = explain_results

        return self

    def evaluate(
        self,
        clf,
        comparisons: list,
        method: str = "all",
        feature_column: str = "Protein",
        verbose: str = False,
        base_score_columns: Optional[list] = None,
    ):
        if base_score_columns is None:
            score_columns = []
        if not "Decoy" in self.row_annotations:
            raise ValueError(
                "No Decoy features found, must call append() on a QuantMatrix first."
            )

        evaluate_results = []

        if isinstance(comparisons, tuple):
            comparisons = [comparisons]

        for comparison in comparisons:

            if base_score_columns:

                score_columns = [
                    f"{score_column}{comparison[0]}-{comparison[1]}"
                    for score_column in base_score_columns
                ]

            else:

                if method == "all":

                    score_columns = [
                        f"DEScore{comparison[0]}-{comparison[1]}",
                        f"Group{comparison[0]}Mean",
                        f"Group{comparison[1]}Mean",
                        f"Group{comparison[0]}Stdev",
                        f"Group{comparison[1]}Stdev",
                        f"Log2FoldChange{comparison[0]}-{comparison[1]}",
                        f"CorrectedPValue{comparison[0]}-{comparison[1]}",
                        f"MeanImportance{comparison[0]}-{comparison[1]}",
                        f"MeanRank{comparison[0]}-{comparison[1]}",
                    ]

                elif method == "ml":
                    score_columns = [
                        f"MeanImportance{comparison[0]}-{comparison[1]}",
                        f"MeanRank{comparison[0]}-{comparison[1]}",
                        f"MedianImportance{comparison[0]}-{comparison[1]}",
                        f"MedianRank{comparison[0]}-{comparison[1]}",
                    ]

                elif method == "deg":

                    score_columns = [
                        f"DEScore{comparison[0]}-{comparison[1]}",
                        f"Group{comparison[0]}Mean",
                        f"Group{comparison[1]}Mean",
                        f"Group{comparison[0]}Stdev",
                        f"Group{comparison[1]}Stdev",
                        f"Log2FoldChange{comparison[0]}-{comparison[1]}",
                        f"CorrectedPValue{comparison[0]}-{comparison[1]}",
                    ]

            X = self.row_annotations[score_columns].copy()

            y = np.where(self.row_annotations["Decoy"] == 0, 1, 0)

            scaler = StandardScaler()

            X[X.columns] = scaler.fit_transform(X[X.columns])

            feature_scores = cross_val_predict(
                clf, X, y, cv=3, method="decision_function"
            )

            scores = cross_val_score(clf, X, y, cv=3)

            feature_score_results = pd.DataFrame(
                {
                    "feature_name": self.row_annotations[feature_column],
                    "label": y,
                    f"{method.capitalize()}FeatureScore{comparison[0]}-{comparison[1]}": feature_scores,
                }
            )

            decoy_counter = DecoyCounter()

            feature_score_results[
                f"{method.capitalize()}FeatureQValue{comparison[0]}-{comparison[1]}"
            ] = decoy_counter.q_values(
                feature_score_results[
                    f"{method.capitalize()}FeatureScore{comparison[0]}-{comparison[1]}"
                ],
                feature_score_results["label"].values,
            )

            self.row_annotations = self.row_annotations.join(
                feature_score_results[
                    [
                        f"{method.capitalize()}FeatureScore{comparison[0]}-{comparison[1]}",
                        f"{method.capitalize()}FeatureQValue{comparison[0]}-{comparison[1]}",
                    ]
                ]
            )

            evaluate_results.append((feature_score_results, scores))

            if verbose:
                print(f"Comparison {comparison[0]}-{comparison[1]}: {scores}")

        self.evaluate_results = evaluate_results

        return self

    def cluster(self, feature_column: str = "Protein", q_value: float = 0.01):
        X, y = self.to_ml(feature_column=feature_column)

        if not "Decoy" in self.row_annotations:
            background = self.append(method="shuffle")

            x_background, _ = QuantMatrix(
                quantification_file=background.to_df()[
                    background.to_df()["Decoy"] == 1
                ].copy(),
                design_matrix_file=self.sample_annotations,
            ).to_ml()

        else:
            x_background = self.quantitative_data[
                (self.quantitative_data.obs["Decoy"] == 1)
            ].X

        clusterer = FeatureClustering(q_value=q_value)

        cluster_ids = clusterer.fit_predict(X, x_background)

        self.row_annotations["FeatureCluster"] = cluster_ids

        self.clusterer = clusterer

        return self

    def enrich(
        self,
        method: str = "overreptest",
        libraries: Optional[list[str]] = None,
        organism: str = "human",
        background: Optional[Union[list[str], str]] = None,
        filter_pvalue: bool = False,
        pvalue_cutoff: float = 0.1,
        pvalue_column: str = "CorrectedPValue2-1",
        filter_importance: bool = False,
        importance_cutoff: float = 0.0,
        importance_column: str = "MeanImportance2-1",
        subset_library: bool = False,
    ):
        """Perform gene set enrichment analysis.

        Args:
           method (str, optional): Enrichment method to use. Options are "enrichr_overreptest" and "overreptest". Defaults to "overreptest".
           libraries (Optional[List[str]], optional): List of gene set libraries. Defaults to None.
           organism (str, optional): Organism for the analysis. Defaults to "human".
           background (Optional[Union[List[str], str]], optional): Background gene set. Defaults to None.
           filter_pvalue (bool, optional): Whether to filter by p-value. Defaults to False.
           pvalue_cutoff (float, optional): P-value cutoff for filtering. Defaults to 0.1.
           pvalue_column (str, optional): Column name for p-values. Defaults to "CorrectedPValue2-1".
           filter_importance (bool, optional): Whether to filter by local perturbation importance value. Defaults to False.
           importance_cutoff (float, optional): Local perturbation importance value cutoff for filtering. Defaults to 0.0.
           importance_column (str, optional): Column name for local perturbation importance values. Defaults to "MeanImportance2-1".
           subset_library (bool, optional): Whether to subset the library. Defaults to False.

        Returns:
           Any: Enrichment result.

        Raises:
           ValueError: If the method is not supported.

        Examples:
            >>> enr = quantified_data.enrich(
            >>>     method="enrichr_overreptest",
            >>>     filter_pvalue=True,
            >>>     pvalue_column="CorrectedPValue2-1",
            >>>     pvalue_cutoff=0.1
            >>> )

        """
        if not self.annotated:
            self.annotate()

        if not libraries:
            libraries = ["GO_Biological_Process_2023"]

        gene_df = pd.DataFrame()

        if filter_pvalue:
            gene_df = self.row_annotations[
                self.row_annotations[pvalue_column] < pvalue_cutoff
            ]

        if filter_importance:
            gene_df = self.row_annotations[
                self.row_annotations[importance_column] > importance_cutoff
            ]

        genes = gene_df["Gene"].to_list()

        if subset_library:
            temp_libraries = []

            for library in libraries:
                go_bp = gp.get_library(name=library, organism=organism)

                gene_set = set(gene_df["Gene"].to_list())

                bio_process_subset = dict()

                for key, value in go_bp.items():
                    for gene in value:
                        if gene in gene_set:
                            bio_process_subset[key] = value

                temp_libraries.append(bio_process_subset)

            libraries = temp_libraries

        enr = None

        if method == "overreptest":
            if background:
                enr = gp.enrich(
                    gene_list=genes,
                    gene_sets=libraries,
                    background=background,
                )

            else:
                enr = gp.enrich(gene_list=genes, gene_sets=libraries)

        elif method == "enrichr_overreptest":
            if background:
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=libraries,
                    organism=organism,
                    background=background,
                )

            else:
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=libraries,
                    organism=organism,
                )

        else:
            raise ValueError(f"Unsupported pathway enrichment method: {method}")

        return enr

    def annotate(self):
        """Annotate proteins with gene names.

        Returns:
            QuantMatrix: The annotated QuantMatrix object.

        Examples:
            >>> quant_matrix.annotate()

        """

        mapping_df = get_genes_from_proteins(self.proteins)

        self.row_annotations = self.row_annotations.join(
            mapping_df.set_index("Protein"), on="Protein", how="left"
        )

        self.row_annotations["Gene"] = self.row_annotations["Gene"].fillna(
            self.row_annotations["Protein"]
        )

        self.annotated = True

        return self

    def predict(
        self,
        classifier,
        scaler: Any = None,
        scale: bool = True,
    ) -> QuantMatrix:
        """Predict labels using a classifier.

        Args:
            classifier: The classifier model to use for prediction.
            scaler (optional): The scaler object to use for data scaling.
            scale (bool): Whether to scale the data before prediction. Defaults to True.

        Returns:
            QuantMatrix: The QuantMatrix object with predicted labels.

        Examples:
            >>> quant_matrix.predict(classifier=clf, scaler=std_scaler)

        """
        X = format_data(self)

        if scale:
            if scaler:
                X = scaler.transform(X)
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

        classifier = Classifier(classifier=classifier)

        self.sample_annotations["Prediction"] = classifier.predict(X)

        return self

    def interpret(
        self,
        classifier,
        scaler: Any = None,
        scale: bool = True,
        downsample_background=False,
    ) -> QuantMatrix:
        """Interpret the model's predictions using local perturbation importance values.

        Args:
            classifier: The classifier model to interpret.
            scaler (optional): The scaler object to use for data scaling.
            scale (bool): Whether to scale the data before interpretation. Defaults to True.
            downsample_background (bool): Whether to downsample background data. Defaults to False.

        Returns:
            QuantMatrix: The QuantMatrix object with local perturbation importance values added to observations.

        Examples:
            >>> quant_matrix.interpret(classifier=clf, scaler=std_scaler)

        """
        X = format_data(self)
        y = encode_labels(self.quantitative_data.var["group"].values)

        if scale:
            if scaler:
                X = scaler.transform(X)
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

        classifier = Classifier(classifier=classifier)

        if downsample_background:
            rus = RandomUnderSampler(random_state=0)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            classifier.interpret(X_resampled)
            self.transformed_data = X_resampled
            self.y_resampled = y_resampled
        else:
            classifier.interpret(X)
            self.transformed_data = X

        self.classifier = classifier
        feature_importances = classifier.feature_importances_.tolist()

        self.quantitative_data.obs["Importance"] = feature_importances

        self.local_feature_importances_ = classifier.explainer.local_explanations

        return self

    def train(
        self,
        classifier,
        scaler: Any = None,
        scale: bool = True,
        validate: bool = True,
        scoring: str = "accuracy",
        num_folds: int = 3,
        random_state: int = 42,
        shuffle: bool = False,
    ) -> TrainResult:
        """Train a classifier on the quantitative data.

        Args:
            classifier: The classifier object or class to use for training.
            scaler (Any): The scaler object to scale the data. Defaults to None.
            scale (bool): Whether to scale the data. Defaults to True.
            validate (bool): Whether to perform cross-validation. Defaults to True.
            scoring (str): The scoring metric for cross-validation. Defaults to "accuracy".
            num_folds (int): The number of folds for cross-validation. Defaults to 3.
            random_state (int): Random seed for reproducibility. Defaults to 42.
            shuffle (bool): Whether to shuffle the data before splitting in cross-validation. Defaults to False.

        Returns:
            TrainResult: The result of the training process, including the trained classifier, scaler, and validation scores.

        Examples:
            >>> result = quant_matrix.train(classifier=RandomForestClassifier(), validate=True)

        """
        X = format_data(self)
        y = encode_labels(self.quantitative_data.var["group"].values)

        if scale:
            if scaler:
                X = scaler.transform(X)
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

        classifier = Classifier(classifier=classifier)

        validation_result = np.array([])

        if validate:
            cv = StratifiedKFold(num_folds, shuffle=shuffle, random_state=random_state)
            validation_result = cross_val_score(
                classifier, X, y, scoring=scoring, cv=cv
            )

        classifier.fit(X, y)

        return TrainResult(classifier, scaler, validation_result)

    def optimize(
        self,
        classifier,
        param_search_method: str,
        param_grid: dict,
        scaler: Any = None,
        scale: bool = True,
        threads: int = 1,
        random_state: int = 42,
        folds: int = 3,
        verbose: Union[bool, int] = False,
        **kwargs: Union[dict, int, str, bool],
    ) -> ParamSearchResult:
        """Optimize hyperparameters of a classifier using different search methods.

        Args:
            classifier: The classifier object or class to optimize.
            param_search_method (str): The parameter search method to use ("genetic" or "random").
            param_grid (dict): The parameter grid to search over.
            scaler (Any): The scaler object to scale the data. Defaults to None.
            scale (bool): Whether to scale the data. Defaults to True.
            threads (int): The number of threads to use for optimization. Defaults to 1.
            random_state (int): Random seed for reproducibility. Defaults to 42.
            folds (int): The number of folds for cross-validation. Defaults to 3.
            verbose (Union[bool, int]): Verbosity level. Defaults to False.
            **kwargs: Additional keyword arguments specific to each search method.

        Returns:
            ParamSearchResult: The result of the parameter search, including the best estimator and parameter populations.

        Examples:
            >>> param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
            >>> result = quant_matrix.optimize(classifier=DecisionTreeClassifier(), param_search_method='random', param_grid=param_grid, verbose=True)
            >>> result.best_estimator_
            DecisionTreeClassifier(max_depth=5, min_samples_split=10)
        """
        X = format_data(self)
        y = encode_labels(self.quantitative_data.var["group"].values)

        if scale:
            if scaler:
                X = scaler.transform(X)
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

        result = None

        if param_search_method == "genetic":
            gas = GeneticAlgorithmSearch(
                classifier,
                param_grid=param_grid,
                threads=threads,
                folds=folds,
                n_survive=kwargs.get("n_survive", 5),
                pop_size=kwargs.get("pop_size", 10),
                n_generations=kwargs.get("n_generations", 20),
                verbose=verbose,
                random_state=kwargs.get("random_state", None),
                shuffle=kwargs.get("shuffle", False),
            )
            parameter_populations = gas.fit(X, y)

            result = ParamSearchResult(
                classifier=gas.best_estimator_,
                result=parameter_populations,
            )

        elif param_search_method == "random":
            randomized_search = RandomizedSearch(
                classifier,
                param_grid=param_grid,
                folds=folds,
                random_state=random_state,
                n_iter=kwargs.get("n_iter", 30),
                n_jobs=threads,
                scoring=kwargs.get("scoring", "accuracy"),
                verbose=verbose,
            )

            result = randomized_search.fit(X, y)

        return result

    def plot(
        self,
        plot_type: str,
        save: bool = False,
        fig: matplotlib.figure.Figure = None,
        ax: Union[list, matplotlib.axes.Axes] = None,
        **kwargs: Union[
            np.ndarray,
            int,
            list,
            str,
        ],
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Generate plots based on specified plot type.

        Args:
            plot_type (str): The type of plot to generate. Possible values are:
                - "importance_summary": Local perturbation importance summary plot.
                - "rfe_pca": Recursive Feature Elimination (RFE) with Principal Component Analysis (PCA) plot.
            save (bool): Whether to save the plot. Defaults to False.
            fig (matplotlib.figure.Figure): The matplotlib figure object. Defaults to None.
            ax (Union[list, matplotlib.axes.Axes]): The list of matplotlib axes objects or a single axes object. Defaults to None.
            **kwargs: Additional keyword arguments specific to each plot type.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The matplotlib figure and axes objects.

        Raises:
            ValueError: If an unsupported plot type is provided.

        Examples:
            >>> fig, ax = quant_matrix.plot(plot_type='importance_summary', save=True, n_display=10)
        """

        if plot_type == "importance_summary":
            try:
                getattr(self, "feature_importances_")
            except AttributeError:
                print("Local perturbation importance values have not been generated")
            cmap = kwargs.get(
                "cmap",
                [
                    "#ff4800",
                    "#ff4040",
                    "#a836ff",
                    "#405cff",
                    "#05c9fa",
                ],
            )

            order_by = kwargs.get("order_by", "importance")

            fig, ax = ImportancePlot(
                fig=fig,
                ax=ax,
                feature_importances=self.local_feature_importances_,
                X=self.transformed_data,
                qm=self,
                cmap=cmap,
                n_display=kwargs.get("n_display", 5),
                jitter=kwargs.get("jitter", 0.1),
                alpha=kwargs.get("alpha", 0.75),
                n_bins=kwargs.get("n_bins", 100),
                feature_column=kwargs.get("feature_column", "Protein"),
                order_by=order_by,
            ).plot()

        if plot_type == "rfe_pca":
            cmap = kwargs.get("cmap", "coolwarm")
            cutoffs = list(kwargs.get("cutoffs", [100, 50, 10]))
            fig, ax = RFEPCA(
                fig=fig, axs=ax, qm=self, cutoffs=cutoffs, cmap=cmap
            ).plot()

        if save:
            filepath = str(kwargs.get("filepath", f"{plot_type}.png"))
            dpi = int(kwargs.get("dpi", 300))
            matplotlib.pyplot.savefig(filepath, dpi=dpi)

        return fig, ax

    def detect(self) -> None:
        """Not implemented

        Detect outliers in the samples
        """

        pass

    def write(self, file_path: str) -> None:
        """Write the QuantMatrix to a tab-separated file.

        Args:
            file_path (str): The path where the file will be saved.

        Returns:
            None

        Examples:
            >>> filename = "protein.tsv"
            >>> quant_matrix.write(filename)
        """

        self.to_df().to_csv(file_path, sep="\t", index=False)

    def to_df(self) -> pd.DataFrame:
        """Convert the QuantMatrix object to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame representation of the QuantMatrix.

        Examples:
            >>> quant_matrix.to_df()

        """

        quant_data = self.quantitative_data[self.row_annotations.index, :].to_df()

        merged = pd.concat([self.row_annotations, quant_data], axis=1)

        return merged

    def to_ml(
        self,
        feature_column: str = "Protein",
        label_column: str = "group",
        comparison: tuple = (1, 2),
    ) -> tuple[Any, Any]:
        """Converts the QuantMatrix object to features and labels for machine learning.

        Args:
            feature_column (str, optional): The column to use as features. Defaults to "Protein".
            label_column (str, optional): The column to use as labels. Defaults to "group".
            comparison (tuple, optional): The comparison groups. Defaults to (1, 2).

        Returns:
            tuple[Any, Any]: A tuple containing features and labels.

        Examples:
            >>> features, labels = quant_matrix.to_ml()
        """
        qm_df = self.to_df()

        samples = self.sample_annotations[
            self.sample_annotations[label_column].isin(comparison)
        ]["sample"].to_list()

        transposed_features = qm_df.set_index(feature_column)[samples].T

        sample_annotations = self.sample_annotations.copy()

        sample_annotations_subset = sample_annotations[
            sample_annotations[label_column].isin(comparison)
        ].copy()

        encoder = LabelEncoder()

        sample_annotations_subset["label"] = encoder.fit_transform(
            sample_annotations_subset[label_column]
        )

        combined = transposed_features.join(
            sample_annotations_subset[["sample", "label"]].set_index("sample"),
            how="left",
        )

        return combined.loc[:, combined.columns != "label"], combined[["label"]]
