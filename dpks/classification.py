from typing import TYPE_CHECKING, Any
import xgboost
from sklearn.model_selection import cross_val_score
import numpy as np
import shap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

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
    def estimator_(self):
        return self.classifier.classifier


class Classifier(BaseEstimator, ClassifierMixin):
    X: np.array
    y: np.array
    shap_values: list
    shap_algorithm: str
    mean_importance: list
    use_sample_weight: bool

    def __init__(self, classifier, shap_algorithm: str = "auto", use_sample_weight: bool = True):
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
        self.use_sample_weight = use_sample_weight

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.use_sample_weight:
            sample_weights = class_weight.compute_sample_weight(
                class_weight="balanced",
                y=y
            )
            self.classifier.fit(X, y, sample_weight=sample_weights)

        else:

            self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def cross_validation(self, X, y, k_folds: int = 5):
        self.scores = cross_val_score(self.classifier, X, y, cv=k_folds)

    def interpret(self, X):
        self.X = X


        if self.shap_algorithm == "permutation":
            explainer = shap.Explainer(
                self.classifier.predict, X, algorithm=self.shap_algorithm
            )

            self.shap_values = explainer(X, max_evals=2 * X.shape[1] + 1).values

        elif (
            self.shap_algorithm == "tree"
            or self.shap_algorithm == "auto"
            or self.shap_algorithm == "partition"
        ):
            explainer = shap.Explainer(self.classifier, algorithm=self.shap_algorithm)
            self.shap_values = explainer.shap_values(X)
        elif self.shap_algorithm == "linear":
            explainer = shap.KernelExplainer(
                self.classifier.predict, data=X, algorithm=self.shap_algorithm
            )
            self.shap_values = explainer.shap_values(X)

        elif self.shap_algorithm == "predict_proba":

            explainer = shap.KernelExplainer(
                self.classifier.predict_proba, data=shap.sample(X, 20), algorithm=self.shap_algorithm
            )
            self.shap_values = explainer.shap_values(X)

        if isinstance(self.shap_values, list):
            self.shap_values = np.swapaxes(np.array(self.shap_values), 1, 2)

        self.mean_importance = np.mean(abs(self.shap_values), axis=0)

    @property
    def feature_importances_(self):
        self.interpret(self.X)
        return self.mean_importance
    
    @property
    def classes_(self):
        return np.unique(self.y)


def encode_labels(labels: np.ndarray) -> np.ndarray:
    encoder = LabelEncoder()

    return encoder.fit_transform(labels)


def format_data(quant_matrix: QuantMatrix) -> np.ndarray:
    X = quant_matrix.quantitative_data.X.copy().transpose()

    return np.nan_to_num(X, copy=True, nan=0.0)
