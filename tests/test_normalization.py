#!/usr/bin/env python3
"""Tests for the normalization module"""

import pytest
import dpks.quant_matrix as quant
from dpks.normalization import NormalizationMethod


@pytest.mark.normalization()
def test_normalization_mean(paths, quant_matrix):
    """normalization mean"""

    assert type(quant_matrix) == quant.QuantMatrix

    normalized_peptide_quantities = quant_matrix.normalize(
        method=NormalizationMethod.MEAN
    )

    assert type(normalized_peptide_quantities) == quant.QuantMatrix
