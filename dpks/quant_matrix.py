"""quant_matrix module

instanciate a quant matrix:

>>> from dpks.quant_matrix import QuantMatrix
>>> quant_matrix = QuantMatrix( quantification_file="tests/input_files/minimal_matrix.tsv", design_matrix_file="tests/input_files/minimal_design_matrix.tsv")

"""
from __future__ import annotations

from sklearn.feature_selection import RFECV, RFE

from dpks.annotate_proteins import get_protein_labels
from typing import Union, List

import numpy as np
import pandas as pd  # type: ignore
import anndata as ad  # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib

from dpks.normalization import (
    TicNormalization,
    MedianNormalization,
    MeanNormalization,
    Log2Normalization,
    NormalizationMethod,
    RTSlidingWindowNormalization,
)
from dpks.scaling import (
    ScalingMethod,
    ZScoreScaling,
    MinMaxScaling,
    AbsMaxScaling,
)
from dpks.imputer import (
    ImputerMethod,
    UniformRangeImputer,
    UniformPercentileImputer,
)
from dpks.plot import Plot, SHAPPlot
from dpks.quantification import TopN, MaxLFQ
from dpks.differential_testing import DifferentialTest
from dpks.classification import Classifier

from dpks.parsers import parse_diann


class QuantMatrix:
    """holds a quantitative matrix and a design matrix, exposes an API to manipulate the quantitative matrix"""

    quantification_file_path: Union[str, pd.DataFrame]
    design_matrix_file: Union[str, pd.DataFrame]
    num_rows: int
    num_samples: int
    quantitative_data: ad.AnnData
    selector: Union[RFECV, RFE]

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

        filtered_data = self.quantitative_data[
            (self.quantitative_data.obs["PeptideQValue"] <= peptide_q_value)
            & (self.quantitative_data.obs["ProteinQValue"] <= protein_q_value)
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
            top_n = int(kwargs.get("top_n", 1))

            quantifications = TopN(top_n=top_n).quantify(self)

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

    def compare_groups(
        self,
        method: str,
        comparisons: list,
        min_samples_per_group: int = 2,
        level: str = "protein",
        multiple_testing_correction_method: str = "fdr_tsbh",
    ) -> QuantMatrix:
        """compare groups by differential testing

        >>> isinstance(quant_matrix.compare_groups(method="linregress", group_a=4, group_b=6), QuantMatrix)
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

    def classify(
        self,
        classifier,
        shap_algorithm: str = "auto",
        scale: bool = True,
        rfe_step: int = 1,
        rfe_min_features_to_select: int = 1,
        min_samples_per_group: int = 2,
        run_rfe: bool = True,
    ) -> QuantMatrix:
        identifiers = self.proteins

        quant_copy = self.quantitative_data.copy()
        quant_copy.X[quant_copy.X == 0.0] = np.nan
        drop_indexes = []
        groups = self.quantitative_data.var["group"].unique()

        for group in groups:
            for identifier in identifiers:
                quant_data = quant_copy[
                    self.row_annotations["Protein"] == identifier, :
                ].copy()

                index = int(quant_data.obs.index.to_numpy()[0])

                group_data = quant_data[:, self.get_samples(group=group)].X.copy()
                group_nonan = len(group_data[~np.isnan(group_data)])

                if group_nonan < min_samples_per_group:
                    if index not in drop_indexes:
                        drop_indexes.append(index)

        le = LabelEncoder()
        Y = le.fit_transform(self.quantitative_data.var["group"].values)
        X = self.quantitative_data.X.copy().transpose()
        X = np.delete(X, drop_indexes, 1)
        X = np.nan_to_num(X, copy=True, nan=0.0)

        if scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        self.clf = Classifier(classifier=classifier, shap_algorithm=shap_algorithm)
        self.clf.fit(X, Y)

        shap_values = self.clf.feature_importances_.tolist()

        for index in drop_indexes:
            shap_values.insert(index, np.nan)
        self.quantitative_data.obs["SHAP"] = shap_values

        if run_rfe:
            selector = self.clf.recursive_feature_elimination(
                X, Y, min_features_to_select=rfe_min_features_to_select, step=rfe_step
            )
            feature_rank_values = selector.ranking_.tolist()
            for index in drop_indexes:
                feature_rank_values.insert(index, np.nan)
            self.quantitative_data.obs["FeatureRank"] = feature_rank_values
            self.selector = selector

        return self

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
        fig=None,
        ax=None,
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
                getattr(self.clf, "shap_values")
            except AttributeError:
                print("SHAP values have not been generated")
            n_display = int(kwargs.get("n_display", 5))
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

            fig, ax = SHAPPlot(
                fig=fig,
                ax=ax,
                shap_values=self.clf.shap_values,
                X=self.clf.X,
                qm=self,
                cmap=cmap,
                n_display=n_display,
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
