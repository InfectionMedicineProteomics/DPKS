#!/usr/bin/env python3
"""Tests for the quant_matrix module"""

import pandas
import dpks.quant_matrix as quant
import pytest


@pytest.mark.quant_matrix()
def test_quant_matrix_as_dataframe(quant_matrix):
    """test quant matrix as dataframe"""

    assert type(quant_matrix) == quant.QuantMatrix

    quant_matrix_df = quant_matrix.as_dataframe(level="protein")

    assert type(quant_matrix_df) == pandas.core.frame.DataFrame
