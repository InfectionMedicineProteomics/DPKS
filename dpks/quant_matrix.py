"""quant_matrix module

instanciate a quant matrix:

>>> from dpks.quant_matrix import QuantMatrix
>>> quant_matrix = QuantMatrix( quantification_file="tests/input_files/minimal_matrix.tsv", design_matrix_file="tests/input_files/minimal_design_matrix.tsv")

"""
from __future__ import annotations
from dpks.annotate_proteins import get_protein_labels
from typing import Union, List

import numpy as np
import pandas as pd  # type: ignore
import anndata as ad  # type: ignore

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
    ZscoreScaling,
)
from dpks.imputer import (
    ImputerMethod,
    UniformImputer,
)
from dpks.quantification import TopN, MaxLFQ
from dpks.differential_testing import DifferentialTest


class QuantMatrix:
    """holds a quantitative matrix and a design matrix, exposes an API to manipulate the quantitative matrix"""

    quantification_file_path: Union[str, pd.DataFrame]
    design_matrix_file: Union[str, pd.DataFrame]
    num_rows: int
    num_samples: int
    quantitative_data: ad.AnnData

    def __init__(
        self,
        quantification_file: Union[str, pd.DataFrame],
        design_matrix_file: Union[str, pd.DataFrame],
        build_quant_graph: bool = False,
        annotate_proteins: bool = True,
    ) -> None:
        """init"""

        if isinstance(design_matrix_file, str):

            design_matrix_file = pd.read_csv(design_matrix_file, sep="\t")

            design_matrix_file.columns = map(str.lower, design_matrix_file.columns)

        if isinstance(quantification_file, str):

            quantification_file = pd.read_csv(quantification_file, sep="\t")

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

        self.quantitative_data = ad.AnnData(
            quantitative_data,
            obs=row_obs,
            var=design_matrix_file.copy().set_index(design_matrix_file["sample"]),
            dtype=np.float64,
        )

        if build_quant_graph:

            pass
        
        if annotate_proteins:
            print(self.quantitative_data.obs)
            self.quantitative_data.obs['ProteinLabel'] = get_protein_labels(self.quantitative_data.obs["Protein"].values)


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

            filtered_data = filtered_data[filtered_data.obs["Decoy"] == 0].copy()

        if remove_contaminants:

            filtered_data = filtered_data[
                ~filtered_data.obs["Protein"].str.contains("contam")
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

            base_method = ZscoreScaling()

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
        group_a: int,
        group_b: int,
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
            group_a,
            group_b,
            min_samples_per_group,
            level,
            multiple_testing_correction_method,
        )

        compared_data = differential_test.test(self)

        self.row_annotations = compared_data.row_annotations.copy()

        return self

    def impute(self, method: str, **kwargs: int) -> QuantMatrix:

        """impute missing values

        not implemented

        """

        base_method: ImputerMethod = ImputerMethod()

        if method == "uniform":

            minvalue = int(kwargs.get("minvalue", 0))
            maxvalue = int(kwargs.get("maxvalue", 1))

            base_method = UniformImputer(
                minvalue=minvalue,
                maxvalue=maxvalue,
            )

        self.quantitative_data.X = base_method.fit_transform(self.quantitative_data.X)

        return self

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
