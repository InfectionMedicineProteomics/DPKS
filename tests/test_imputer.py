#!/usr/bin/env python3
"""Testing the imputer module"""

import pytest
from dpks.quant_matrix import QuantMatrix
import pandas


@pytest.mark.imputer()
def test_imputer(quant_matrix):
    """test imputer"""

    assert isinstance(quant_matrix, QuantMatrix)

    imputed_peptide_quantities = quant_matrix.impute(method="random")

    assert isinstance(imputed_peptide_quantities, QuantMatrix)

    imputed_peptide_quantities_df = imputed_peptide_quantities.to_df()

    assert isinstance(imputed_peptide_quantities_df, pandas.core.frame.DataFrame)
