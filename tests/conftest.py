#!/usr/bin/env python

import os
import pytest
from pathlib import Path
from dpks.quant_matrix import QuantMatrix
from dpks.plot import Plot

base_dir = Path(os.getcwd())


@pytest.fixture(scope="session")
def paths(tmpdir_factory):
    """create a dictionary with paths to input data files"""

    paths = {}
    paths["test_base_path"] = Path(tmpdir_factory.mktemp("dpks"))
    paths["design_matrix_path"] = base_dir / Path("tests/input_files/design_matrix.tsv")
    paths["baseline_matrix_path"] = base_dir / Path(
        "tests/input_files/pyprophet_baseline_matrix.tsv"
    )

    yield paths


@pytest.fixture(scope="session")
def quant_matrix(paths):
    """instanciate a quant_matrix"""
    assert paths["baseline_matrix_path"].is_file()
    assert paths["design_matrix_path"].is_file()
    quant_matrix = QuantMatrix(
        str(paths["baseline_matrix_path"]),
        str(paths["design_matrix_path"]),
    )
    yield quant_matrix


@pytest.fixture(scope="session")
def plot_object():
    """create a plot object"""
    yield Plot()
