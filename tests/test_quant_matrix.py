#!/usr/bin/env python3
"""Tests for the quant_matrix module"""

import pandas
from dpks.quant_matrix import QuantMatrix
import pytest


@pytest.mark.quant_matrix()
def test_quant_matrix_as_dataframe(quant_matrix):
    """test quant matrix as dataframe"""

    assert type(quant_matrix) == QuantMatrix

    quant_matrix_df = quant_matrix.to_df()

    assert type(quant_matrix_df) == pandas.core.frame.DataFrame
