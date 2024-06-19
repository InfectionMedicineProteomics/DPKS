from typing import TYPE_CHECKING, Any
import xgboost
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from dpks.inspection import FeatureImportance

if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class TrainResult:
    """
    Class to store the results of a training process.

    Parameters:
    - classifier: Trained classifier.
    - scaler: Scaler used during training.
    - validation_results: Results from the validation process.

    Attributes:
    - classifier: Trained classifier.
    - scaler: Scaler used during training.
    - validation_results: Results from the validation process.
    - estimator_: Alias for the trained classifier.

    """

    def __init__(self, classifier, scaler, validation_results):
        """
        Initialize TrainResult with classifier, scaler, and validation results.

        Parameters:
        - classifier: Trained classifier.
        - scaler: Scaler used during training.
        - validation_results: Results from the validation process.

        Returns:
        None
        """
        self.classifier = classifier
        self.scaler = scaler
        self.validation_results = validation_results

    @property
    def estimator_(self):
        """
        Alias for the trained classifier.

        Returns:
        Any: Trained classifier.
        """
        return self.classifier.classifier


class Classifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper class for classifiers with added functionality.

    Parameters:
    - classifier: Base classifier or string identifier for XGBoost.
    - use_sample_weight (bool): Whether to use sample weights during training.

    Methods:
    - fit(X, y): Fit the classifier to the data.
    - predict(X): Predict labels for input data.
    - cross_validation(X, y, k_folds): Perform cross-validation and store scores.
    - interpret(X): Interpret the model using local purturbation importance values.
    - feature_importances_: Get feature importances based on local purturbation importance values.

    """

    X: np.array
    y: np.array
    mean_importance: np.ndarray
    use_sample_weight: bool

    def __init__(
        self, classifier, use_sample_weight: bool = True
    ):
        """
        Initialize the Classifier with a base classifier and optional parameters.

        Parameters:
        - classifier: Base classifier or string identifier for XGBoost.
        - use_sample_weight: Whether to use sample weights during training.

        Returns:
        None
        """
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
        self.use_sample_weight = use_sample_weight

    def fit(self, X, y):
        """
        Fit the classifier to the data.

        Parameters:
        - X: Input data.
        - y: Target labels.

        Returns:
        Classifier: The fitted classifier.
        """
        self.X = X
        self.y = y
        if self.use_sample_weight:
            sample_weights = class_weight.compute_sample_weight(
                class_weight="balanced", y=y
            )
            self.classifier.fit(X, y, sample_weight=sample_weights)

        else:
            self.classifier.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict labels for input data.

        Parameters:
        - X: Input data.

        Returns:
        np.array: Predicted labels.
        """

        return self.classifier.predict(X)

    def cross_validation(self, X, y, k_folds: int = 5):
        """
        Perform cross-validation and store scores.

        Parameters:
        - X: Input data.
        - y: Target labels.
        - k_folds: Number of folds for cross-validation.

        Returns:
        None
        """
        self.scores = cross_val_score(self.classifier, X, y, cv=k_folds)

    def interpret(self, X):
        """
        Interpret the model.

        Parameters:
        - X: Input data.

        Returns:
        None
        """
        self.X = X

        explainer = FeatureImportance(
            n_iterations=3,
            feature_names=X.columns.values
        )

        explainer.fit(self.classifier, X.values)

        self.explainer = explainer
        self.mean_importance = explainer.global_explanations

    @property
    def feature_importances_(self):
        """
        Get feature importances.

        Returns:
        np.array: Feature importances.
        """
        self.interpret(self.X)
        return self.mean_importance

    @property
    def classes_(self):
        return np.unique(self.y)


def encode_labels(labels: np.ndarray) -> np.ndarray:
    """
    Encode labels using LabelEncoder.

    Parameters:
    - labels: Array of labels.

    Returns:
    np.ndarray: Encoded labels.
    """
    encoder = LabelEncoder()

    return encoder.fit_transform(labels)


def format_data(quant_matrix: QuantMatrix) -> np.ndarray:
    """
    Format quantitative matrix data.

    Parameters:
    - quant_matrix: QuantMatrix instance.

    Returns:
    np.ndarray: Formatted data.
    """
    X = quant_matrix.quantitative_data.X.copy().transpose()

    return np.nan_to_num(X, copy=True, nan=0.0)
