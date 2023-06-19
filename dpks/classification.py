from typing import TYPE_CHECKING, Any
import xgboost
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import shap
from sklearn.feature_selection import RFECV, RFE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class TrainResult:
    def __init__(self, classifier, scaler, validation_results):
        self.classifier = classifier
        self.scaler = scaler
        self.validation_results = validation_results

    @property
    def esimator_(self):
        return self.classifier.classifier


class Classifier(BaseEstimator, ClassifierMixin):
    X: np.array
    y: np.array
    shap_values: list
    shap_algorithm: str
    mean_importance: list

    def __init__(
        self,
        classifier,
        shap_algorithm: str = "auto",
    ):
        if isinstance(classifier, str):
            if classifier == "xgboost":
                self.classifier = xgboost.XGBClassifier(
                    max_depth=30,
                    eval_metric="logloss",
                    verbosity=0,
                )
        else:
            fit_method = getattr(classifier, "fit", None)
            predict_method = getattr(classifier, "predict", None)
            if callable(fit_method) and callable(predict_method):
                self.classifier = classifier
            else:
                raise ValueError(
                    "The classifier does not have a fit and/or predict method"
                )
        self.shap_algorithm = shap_algorithm

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classifier.fit(X, y)
        self.interpret(X)
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def cross_validation(self, X, y, k_folds: int = 5):
        self.scores = cross_val_score(self.classifier, X, y, cv=k_folds)

    def interpret(self, X):
        if self.shap_algorithm == "permutation":
            explainer = shap.Explainer(
                self.classifier.predict, X, algorithm=self.shap_algorithm
            )
            self.shap_values = explainer(X, max_evals=2 * X.shape[1] + 1)
        elif (
            self.shap_algorithm == "tree"
            or self.shap_algorithm == "auto"
            or self.shap_algorithm == "linear"
            or self.shap_algorithm == "partition"
        ):
            explainer = shap.Explainer(self.classifier, algorithm=self.shap_algorithm)
            self.shap_values = explainer.shap_values(X)

        self.mean_importance = np.mean(abs(self.shap_values), axis=0)

    @property
    def feature_importances_(self):
        return self.mean_importance


def encode_labels(labels: np.ndarray) -> np.ndarray:
    encoder = LabelEncoder()

    return encoder.fit_transform(labels)


def format_data(quant_matrix: QuantMatrix) -> np.ndarray:
    X = quant_matrix.quantitative_data.X.copy().transpose()

    return np.nan_to_num(X, copy=True, nan=0.0)
