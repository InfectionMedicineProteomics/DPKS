#!/usr/bin/env python3
"""Tests for the differential_testing module"""

import pytest
from dpks.quant_matrix import QuantMatrix


@pytest.mark.differential_testing()
def test_differential_testing(paths):
    """test the differential_testing object"""

    quant_matrix = QuantMatrix(
        quantification_file=str(paths["de_matrix_path"]),
        design_matrix_file=str(paths["de_design_matrix_path"]),
    )
    compared_data = (
        quant_matrix.filter()
        .normalize(
            method="mean",
            #                log_transform=True,
            #                use_rt_sliding_window_filter=True,
            #                window_length=100,
            #                stride=1
        )
        .quantify(method="top_n", top_n=1)
        .compare(
            method="linregress",
            group_a=4,
            group_b=6,
            min_samples_per_group=2,
            level="protein",
            multiple_testing_correction_method="fdr_tsbh",
        )
    )

    assert isinstance(compared_data, QuantMatrix)
