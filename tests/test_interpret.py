from dpks.quant_matrix import QuantMatrix
import pytest
import xgboost
from sklearn.svm import SVC
from sklearn import tree


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


def test_xgb(quantified_data):
    clf = xgboost.XGBClassifier()
    trained_classifier = quantified_data.train(clf)
    quantified_data.interpret(trained_classifier.esimator_, trained_classifier.scaler)


def test_decision_tree(quantified_data):
    clf = tree.DecisionTreeClassifier()
    trained_classifier = quantified_data.train(clf, shap_algorithm="tree")
    quantified_data.interpret(
        trained_classifier.esimator_,
        trained_classifier.scaler,
        shap_algorithm="tree",
    )


def test_svm(quantified_data):
    clf = SVC()
    trained_classifier = quantified_data.train(clf, shap_algorithm="permutation")
    quantified_data.interpret(
        trained_classifier.esimator_,
        trained_classifier.scaler,
        shap_algorithm="permutation",
    )
