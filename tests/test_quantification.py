#!/usr/bin/env python3
"""Tests for the quantification module"""

import pytest
from dpks.quant_matrix import QuantMatrix
from pathlib import Path
import pandas


@pytest.mark.quantification()
def test_quantification_protein(paths, quant_matrix):
    """quantification protein"""

    protein_quantities = quant_matrix.quantify(
        method="top_n",
        top_n=1,
    )

    assert type(protein_quantities) == QuantMatrix

    protein_quantities_df = protein_quantities.to_df()

    assert type(protein_quantities_df) == pandas.core.frame.DataFrame

    assert protein_quantities_df.shape == (570, 264)

    output_file = paths["test_base_path"] / Path("norm_prot_quant.tsv")
    protein_quantities_df.to_csv(output_file, sep="\t", index=False)

    assert output_file.is_file()
