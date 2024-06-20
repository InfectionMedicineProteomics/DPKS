from dpks.quant_matrix import QuantMatrix
import pytest
from sklearn.linear_model import LogisticRegression

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


def test_xgb(quantified_data: QuantMatrix):

    clf = LogisticRegression()
    quantified_data.explain(clf, comparisons=(1, 2), n_iterations=10)
