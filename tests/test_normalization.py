#!/usr/bin/env python3
"""Tests for the normalization module"""

import pytest
from dpks.quant_matrix import QuantMatrix


@pytest.mark.normalization()
def test_normalization_mean(quant_matrix):
    """normalization mean"""

    assert isinstance(quant_matrix, QuantMatrix)

    normalized_peptide_quantities = quant_matrix.normalize(method="mean")

    assert isinstance(normalized_peptide_quantities, QuantMatrix)


@pytest.mark.normalization()
def test_normalization_tic(quant_matrix):
    """normalization tic"""

    assert isinstance(quant_matrix, QuantMatrix)

    normalized_peptide_quantities = quant_matrix.normalize(method="tic")

    assert isinstance(normalized_peptide_quantities, QuantMatrix)


@pytest.mark.normalization()
def test_normalization_median(quant_matrix):
    """normalization median"""

    assert isinstance(quant_matrix, QuantMatrix)

    normalized_peptide_quantities = quant_matrix.normalize(method="median")

    assert isinstance(normalized_peptide_quantities, QuantMatrix)


@pytest.mark.normalization()
def test_normalization_rt_median(quant_matrix):
    """normalization median"""

    assert isinstance(quant_matrix, QuantMatrix)

    normalized_peptide_quantities = quant_matrix.normalize(
        method="mean",
        log_transform=True,
        use_rt_sliding_window_filter=True,
        minimum_data_points=100,
        stride=5,
        use_overlapping_windows=True,
        rt_unit="seconds",
    )

    assert isinstance(normalized_peptide_quantities, QuantMatrix)
