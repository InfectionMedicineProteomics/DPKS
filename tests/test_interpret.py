from dpks.quant_matrix import QuantMatrix
import pytest
import xgboost
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


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
    clf = xgboost.XGBClassifier()
    quantified_data.explain(clf, comparisons=(1, 2), n_iterations=10)


def test_decision_tree(quantified_data):
    clf = tree.DecisionTreeClassifier()
    quantified_data.explain(clf, comparisons=(1, 2), n_iterations=10)


def test_knn(quantified_data):
    clf = KNeighborsClassifier()
    quantified_data.explain(clf, comparisons=(1, 2), n_iterations=10)


def test_svm(quantified_data):
    clf = SVC()
    quantified_data.explain(clf, comparisons=(1, 2), n_iterations=10)
