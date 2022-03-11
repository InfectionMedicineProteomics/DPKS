#!/usr/bin/env python3
"""Tests for the quantification module"""

import pytest
from dpks.quantification import ProteinQuantificationMethod
import dpks.quant_matrix as quant
from pathlib import Path
import pandas


@pytest.mark.quantification()
def test_quantification_protein(paths, quant_matrix):
    """quantification protein"""

    protein_quantities = quant_matrix.quantify(
        method=ProteinQuantificationMethod.TOP_N_PRECURSORS,
        top_n=1,
        protein_grouping="proteotypic",
    )

    assert type(protein_quantities) == quant.QuantMatrix

    protein_quantities_df = protein_quantities.as_dataframe(level="protein")

    assert type(protein_quantities_df) == pandas.core.frame.DataFrame

    assert protein_quantities_df.shape == (570, 264)

    output_file = paths["test_base_path"] / Path("norm_prot_quant.tsv")
    protein_quantities_df.to_csv(output_file, sep="\t", index=False)

    assert output_file.is_file()
