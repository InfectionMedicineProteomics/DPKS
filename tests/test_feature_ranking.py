from dpks.quant_matrix import QuantMatrix
from dpks.feature_selection import BoostrapRFE
import pytest
import xgboost
import pandas as pd


@pytest.fixture
def quantified_data(paths):
    qm = QuantMatrix(
        quantification_file=str(paths["sepsis_matrix_path"]),
        design_matrix_file=str(paths["sepsis_design_path"]),
    )
    quantified_data = (
        qm.normalize(
            method="mean",
        )
        .quantify(method="top_n", summarization_method="mean")
        .impute(method="constant", constant=0)
    )
    return quantified_data


def test_rfe(quantified_data: QuantMatrix):
    """
    Not implemented.
    """
    X,y = quantified_data.to_ml()
    clf = xgboost.XGBClassifier()
    bootstrap_rfe = BoostrapRFE(step=20, downsample_rate=1)

    bootstrap_rfe.fit(X,y, clf)
    feature_ranks = bootstrap_rfe.get_ranks()
    
    assert isinstance(feature_ranks, pd.DataFrame)
