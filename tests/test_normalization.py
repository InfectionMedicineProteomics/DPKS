#!/usr/bin/env python3
"""Tests for the normalization module"""

import pytest
from dpks.quant_matrix import QuantMatrix


@pytest.mark.normalization()
def test_normalization_mean(paths, quant_matrix):
    """normalization mean"""

    assert type(quant_matrix) == QuantMatrix

    normalized_peptide_quantities = quant_matrix.normalize(method="mean")

    assert type(normalized_peptide_quantities) == QuantMatrix
