#!/usr/bin/env python3
"""Tests for the scaling module"""

import pytest
from dpks.quant_matrix import QuantMatrix
import pandas


@pytest.mark.scaling()
def test_scaling_zscore(quant_matrix):
    """zscore scaling"""

    assert isinstance(quant_matrix, QuantMatrix)

    scaled_peptide_quantities = quant_matrix.scale(method="zscore")

    assert isinstance(scaled_peptide_quantities, QuantMatrix)

    scaled_peptide_quantities_df = scaled_peptide_quantities.to_df()

    assert isinstance(scaled_peptide_quantities_df, pandas.core.frame.DataFrame)
