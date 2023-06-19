from dpks.quant_matrix import QuantMatrix
import pytest
import xgboost
from sklearn.svm import SVC


@pytest.fixture
def quantified_data(paths):
    qm = QuantMatrix(
        quantification_file=str(paths["sepsis_matrix_path"]),
        design_matrix_file=str(paths["sepsis_design_path"]),
    )
    quantified_data = qm.normalize(
        method="mean",
    ).quantify(method="top_n", summarization_method="mean")
    return quantified_data


def test_rfe(quantified_data: QuantMatrix):
    clf = xgboost.XGBClassifier()
    quantified_data.rank(clf, rfe_step=100, verbose=True, threads=4)
