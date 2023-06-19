"""quant_matrix module

instanciate a quant matrix:

>>> from dpks.quant_matrix import QuantMatrix
>>> quant_matrix = QuantMatrix( quantification_file="tests/input_files/minimal_matrix.tsv", design_matrix_file="tests/input_files/minimal_design_matrix.tsv")

"""
from __future__ import annotations

from typing import Union, List, Any

import anndata as ad
from sklearn.model_selection import cross_val_score

from dpks.param_search import GeneticAlgorithmSearch, RandomizedSearch, ParamSearchResult  # type: ignore
import matplotlib
import numpy as np
import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler

from dpks.annotate_proteins import get_protein_labels
from dpks.classification import Classifier, encode_labels, format_data, TrainResult
from dpks.differential_testing import DifferentialTest
from dpks.feature_ranking import FeatureRankerRFE
from dpks.imputer import (
    ImputerMethod,
    UniformRangeImputer,
    UniformPercentileImputer,
)
from dpks.normalization import (
    TicNormalization,
    MedianNormalization,
    MeanNormalization,
    Log2Normalization,
    NormalizationMethod,
    RTSlidingWindowNormalization,
)
from dpks.parsers import parse_diann
from dpks.plot import SHAPPlot, RFEPCA
from dpks.quantification import TopN, MaxLFQ
from dpks.scaling import (
    ScalingMethod,
    ZScoreScaling,
    MinMaxScaling,
    AbsMaxScaling,
)


class QuantMatrix:
    """holds a quantitative matrix and a design matrix, exposes an API to manipulate the quantitative matrix"""

    quantification_file_path: Union[str, pd.DataFrame]
    design_matrix_file: Union[str, pd.DataFrame]
    num_rows: int
    num_samples: int
    quantitative_data: ad.AnnData
    selector: FeatureRankerRFE

    def __init__(
        self,
        quantification_file: Union[str, pd.DataFrame],
        design_matrix_file: Union[str, pd.DataFrame],
        annotation_fasta_file: str = None,
        build_quant_graph: bool = False,
        quant_type: str = "gps",
        diann_qvalue: float = 0.01,
    ) -> None:
        """init"""

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
            dtype=np.float64,
        )

        if build_quant_graph:
            pass

    @property
    def proteins(self) -> List[str]:
        """returns unique list of Proteins.

        >>> sorted(quant_matrix.proteins)[40:42]
        ['sp|G5E8V9|ARFP1_MOUSE', 'sp|O08528|HXK2_MOUSE']

        """

        return list(self.quantitative_data.obs["Protein"].unique())

    @property
    def peptides(self) -> List[str]:
        """returns unique list of PeptideSequences

        >>> sorted(quant_matrix.peptides)[0:2]
        ['AAAAGALAPGPLPDLAAR', 'AAAEGVANLHLDEATGEMVSK']

        """

        return list(self.quantitative_data.obs["PeptideSequence"].unique())

    @property
    def precursors(self) -> List[str]:
        """returns unique list of PrecursorIds

        >>> sorted(quant_matrix.precursors)[0:2]
        ['AAAAGALAPGPLPDLAAR_2', 'AAAEGVANLHLDEATGEMVSK_3']

        """

        self.row_annotations["PrecursorId"] = (
            self.row_annotations["PeptideSequence"]
            + "_"
            + self.row_annotations["Charge"].astype(str)
        )

        return list(self.row_annotations["PrecursorId"].unique())

    @property
    def sample_annotations(self) -> pd.DataFrame:
        """returns list of the sample annotations

        >>> sorted(quant_matrix.sample_annotations)
        ['group', 'sample']

        """

        return self.quantitative_data.var

    @property
    def row_annotations(self) -> pd.DataFrame:
        """returns the row observations

        >>> sorted(quant_matrix.row_annotations)
        ['Charge',
         'Decoy',
         'PeptideQValue',
         'PeptideSequence',
         'PrecursorId',
         'Protein',
         'ProteinQValue',
         'RT']

        """

        return self.quantitative_data.obs

    @row_annotations.setter
    def row_annotations(self, value: pd.DataFrame) -> None:
        self.quantitative_data.obs = value

    def get_samples(self, group: int) -> List[str]:
        """return sample names for wanted group

        >>> sorted(quant_matrix.get_samples(group=4))
        ['AAS_P2009_169',
         'AAS_P2009_178',
         'AAS_P2009_187',
         'AAS_P2009_196',
         'AAS_P2009_205',
         'AAS_P2009_214',
         'AAS_P2009_232',
         'AAS_P2009_241',
         'AAS_P2009_250']

        """

        return list(
            self.sample_annotations[self.sample_annotations["group"] == group]["sample"]
        )

    def filter(
        self,
        peptide_q_value: float = 0.01,
        protein_q_value: float = 0.01,
        remove_decoys: bool = True,
        remove_contaminants: bool = True,
        remove_non_proteotypic: bool = True,
    ) -> QuantMatrix:
        """filter the QuantMatrix

        - removes decoys by default
        - removes contaminants by default
        - filters on peptide_q_value <= 0.01 by default
        - filters on protein_q_value <= 0.01 by default

        >>> print(quant_matrix.to_df().shape)
        (16679, 26)
        >>> print(quant_matrix.filter(peptide_q_value=0.001).to_df().shape)
        (15355, 26)

        """

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
            quantitative_data,
            obs=row_obs,
            var=filtered_data.var,
            dtype=np.float64,
        )

        return self

    def scale(
        self,
        method: str,
    ) -> QuantMatrix:
        base_method: ScalingMethod = ScalingMethod()

        if method == "zscore":
            base_method = ZScoreScaling()

        elif method == "minmax":
            base_method = MinMaxScaling()

        elif method == "absmax":
            base_method = AbsMaxScaling()

        self.quantitative_data.X = base_method.fit_transform(self.quantitative_data.X)

        return self

    def normalize(
        self,
        method: str,
        log_transform: bool = True,
        use_rt_sliding_window_filter: bool = False,
        **kwargs: Union[int, bool, str],
    ) -> QuantMatrix:
        """normalize the QuantMatrix

        - need to specify a method
        - log-transform by default
        - can use a sliding window filter, not turned on by default

        >>> isinstance(quant_matrix.normalize(method="tic"), QuantMatrix)
        True

        """

        base_method: NormalizationMethod = NormalizationMethod()

        if method == "tic":
            base_method = TicNormalization()

        elif method == "median":
            base_method = MedianNormalization()

        elif method == "mean":
            base_method = MeanNormalization()

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

        if log_transform:
            self.quantitative_data.X = Log2Normalization().fit_transform(
                self.quantitative_data.X
            )

        return self

    def quantify(
        self,
        method: str,
        resolve_protein_groups: bool = False,
        **kwargs: Union[int, str],
    ) -> QuantMatrix:
        """calculate protein quantities

        - have to specify a method
        - does not resolve protein groups by default

        >>> quant_matrix.quantify(method="top_n", top_n=1).to_df().shape
        (3738, 19)

        """

        if resolve_protein_groups:
            pass

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

            quantifications = MaxLFQ(
                level=level,
                threads=threads,
                minimum_subgroups=minimum_subgroups,
            ).quantify(self)

            design_matrix = self.quantitative_data.var

            protein_quantifications = QuantMatrix(
                quantifications, design_matrix_file=design_matrix
            )

        return protein_quantifications

    def to_df(self) -> pd.DataFrame:
        """to_df converts the QuantMatrix object to a pandas dataframe

        >>> isinstance(quant_matrix.to_df(), pd.DataFrame)
        True

        """

        quant_data = self.quantitative_data[self.row_annotations.index, :].to_df()

        merged = pd.concat([self.row_annotations, quant_data], axis=1)

        return merged

    def compare(
        self,
        method: str,
        comparisons: list,
        min_samples_per_group: int = 2,
        level: str = "protein",
        multiple_testing_correction_method: str = "fdr_tsbh",
    ) -> QuantMatrix:
        """compare groups by differential testing

        >>> isinstance(quant_matrix.compare(method="linregress", group_a=4, group_b=6), QuantMatrix)
        True

        """

        differential_test = DifferentialTest(
            method,
            comparisons,
            min_samples_per_group,
            level,
            multiple_testing_correction_method,
        )

        compared_data = differential_test.test(self)

        self.row_annotations = compared_data.row_annotations.copy()

        return self

    def rank(
        self,
        classifier,
        scaler: Any = None,
        shap_algorithm: str = "auto",
        scale: bool = True,
        rank_method: str = "rfecv",
        **kwargs: Union[dict, int, str, bool],
    ):
        X = format_data(self)
        y = encode_labels(self.quantitative_data.var["group"].values)

        if scale:
            if scaler:
                X = scaler.transform(X)
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

        verbose = bool(kwargs.get("verbose", False))

        if rank_method == "rfecv":
            rfe_step = int(kwargs.get("rfe_step", 1))
            rfe_min_features_to_select = int(
                kwargs.get("rfe_min_features_to_select", 1)
            )

            k_folds = int(kwargs.get("k_folds", 2))

            threads = int(kwargs.get("threads", 1))

            selector = FeatureRankerRFE(
                min_features_to_select=rfe_min_features_to_select,
                step=rfe_step,
                importance_getter=shap_algorithm,
                scoring="accuracy",
                k_folds=k_folds,
                threads=threads,
                verbose=verbose,
            )

            selector.rank_features(X, y, classifier)

            feature_rank_values = selector.ranking_.tolist()

            self.quantitative_data.obs["FeatureRank"] = feature_rank_values

            self.selector = selector

        return self

    def predict(
        self,
        classifier,
        scaler: Any = None,
        shap_algorithm: str = "auto",
        scale: bool = True,
    ) -> QuantMatrix:
        X = format_data(self)

        if scale:
            if scaler:
                X = scaler.transform(X)
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

        classifier = Classifier(classifier=classifier, shap_algorithm=shap_algorithm)

        self.sample_annotations["Prediction"] = classifier.predict(X)

        return self

    def interpret(
        self,
        classifier,
        scaler: Any = None,
        shap_algorithm: str = "auto",
        scale: bool = True,
    ) -> QuantMatrix:
        X = format_data(self)

        if scale:
            if scaler:
                X = scaler.transform(X)
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

        classifier = Classifier(classifier=classifier, shap_algorithm=shap_algorithm)

        classifier.interpret(X)

        shap_values = classifier.feature_importances_.tolist()

        self.quantitative_data.obs["SHAP"] = shap_values

        self.shap = classifier.shap_values
        self.transformed_data = X

        return self

    def train(
        self,
        classifier,
        scaler: Any = None,
        shap_algorithm: str = "auto",
        scale: bool = True,
        validate: bool = True,
        scoring: str = "accuracy",
    ) -> TrainResult:
        X = format_data(self)
        y = encode_labels(self.quantitative_data.var["group"].values)

        if scale:
            if scaler:
                X = scaler.transform(X)
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

        classifier = Classifier(classifier=classifier, shap_algorithm=shap_algorithm)

        validation_result = np.array([])

        if validate:
            validation_result = cross_val_score(classifier, X, y, scoring=scoring)

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

    def impute(self, method: str, **kwargs: int) -> QuantMatrix:
        """impute missing values"""

        base_method: ImputerMethod = ImputerMethod()

        if method == "uniform_percentile":
            percentile = float(kwargs.get("percentile", 0.1))

            base_method = UniformPercentileImputer(percentile=percentile)

        elif method == "uniform_range":
            maxvalue = int(kwargs.get("maxvalue", 1))
            minvalue = int(kwargs.get("minvalue", 0))

            base_method = UniformRangeImputer(maxvalue=maxvalue, minvalue=minvalue)

        self.quantitative_data.X = base_method.fit_transform(self.quantitative_data.X)

        return self

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
        """generate plots"""

        if plot_type == "shap_summary":
            try:
                getattr(self, "shap")
            except AttributeError:
                print("SHAP values have not been generated")
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

            order_by = kwargs.get("order_by", "shap")

            fig, ax = SHAPPlot(
                fig=fig,
                ax=ax,
                shap_values=self.shap,
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

    def outlier_detection(self) -> None:
        """detect outlies


        not implemented

        """

        pass

    def flag_bad_runs(self) -> None:
        """flag bad runs

        not implemented"""

        pass

    def write(self, file_path: str = "") -> None:
        """write the QuantMatrix to a tab-separated file

        >>> from pathlib import Path
        >>> filename = Path("protein.tsv")
        >>> quant_matrix.write(str(filename))
        >>>> filename.is_file()
        True

        """

        self.to_df().to_csv(file_path, sep="\t", index=False)
