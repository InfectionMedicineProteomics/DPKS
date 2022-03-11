#!/usr/bin/env python

import os
import pytest
from pathlib import Path
import dpks.quant_matrix as quant

base_dir = Path(os.getcwd())


@pytest.fixture(scope="session")
def paths(tmpdir_factory):
    """create a dictionary with paths to input data files"""

    paths = {}
    paths["test_base_path"] = Path(tmpdir_factory.mktemp("dpks"))
    paths["design_matrix_path"] = base_dir / Path("tests/input_files/design_matrix.tsv")
    paths["baseline_matrix_path"] = base_dir / Path(
        "tests/input_files/pyprophet_baseline_matrix.csv"
    )

    yield paths


@pytest.fixture(scope="session")
def quant_matrix(paths):
    """instanciate a quant_matrix"""
    assert paths["baseline_matrix_path"].is_file()
    assert paths["design_matrix_path"].is_file()
    quant_matrix = quant.create_quant_matrix(
        str(paths["baseline_matrix_path"]),
        str(paths["design_matrix_path"]),
        quant_type="precursor",
    )
    yield quant_matrix
