from dpks.quant_matrix import QuantMatrix
import pytest
import xgboost
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


@pytest.fixture
def quantified_data(paths):

    clf = xgboost.XGBClassifier()

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


def test_append(quantified_data: QuantMatrix):
    
    quantified_data.append(method="shuffle", feature_column="Protein")
