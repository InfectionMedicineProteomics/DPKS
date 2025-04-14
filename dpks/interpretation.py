from typing import Optional, List
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from sklearn.utils import resample, class_weight

from imblearn.under_sampling import RandomUnderSampler
from kneed import KneeLocator


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
    - classifier: Base classifier.
    - use_sample_weight (bool): Whether to use sample weights during training.

    Methods:
    - fit(X, y): Fit the classifier to the data.
    - predict(X): Predict labels for input data.
    - cross_validation(X, y, k_folds): Perform cross-validation and store scores.
    - interpret(X): Interpret the model using local perturbation importance values.
    - feature_importances_: Get feature importances based on local perturbation importance values.

    """

    X: np.array
    y: np.array
    mean_importance: np.ndarray
    use_sample_weight: bool

    def __init__(
        self,
        classifier,
        use_sample_weight: bool = True,
        shuffle_iterations: int = 3,
        feature_names: list[str] = None,
    ):
        """
        Initialize the Classifier with a base classifier and optional parameters.

        Parameters:
        - classifier: Base classifier.
        - use_sample_weight: Whether to use sample weights during training.

        Returns:
        None
        """
        if isinstance(classifier, str):
            raise ValueError("Must pass in an sklearn compatible classifier")
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
        self.shuffle_iterations = shuffle_iterations
        self.feature_names = feature_names

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilityies for input data.

        Parameters:
        - X: Input data.

        Returns:
        np.array: Predicted probabilities.
        """

        return self.classifier.predict_proba(X)

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
            n_iterations=self.shuffle_iterations, feature_names=self.feature_names
        )

        explainer.fit(self.classifier, X)

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


class FeatureImportance:
    all_predictions: np.ndarray
    feature_names: list[str]

    def __init__(
        self,
        n_iterations: int = 3,
        feature_names: list[str] = None,
        problem_type: str = "classification",
    ):
        self.n_iterations = n_iterations
        self.feature_names = feature_names
        self.problem_type = problem_type

    def fit(self, clf, X) -> None:
        decision_function = getattr(clf, "decision_function", None)

        if self.problem_type == "classification":

            if callable(decision_function):
                all_predictions = clf.decision_function(X)
            else:
                all_predictions = clf.predict_proba(X)[:, 1]

        elif self.problem_type == "regression":

            all_predictions = clf.predict(X)

        global_explanations = []

        local_explanations = []

        for i in range(X.shape[1]):
            feature_slice = X[:, i].copy()

            X_copy = X.copy()

            losses = []

            for _ in range(self.n_iterations):
                np.random.shuffle(feature_slice)

                X_copy[:, i] = feature_slice

                if self.problem_type == "classification":

                    if callable(decision_function):
                        new_predictions = clf.decision_function(X_copy)
                    else:
                        new_predictions = clf.predict_proba(X_copy)[:, 1]

                elif self.problem_type == "regression":

                    new_predictions = clf.predict(X_copy)

                local_lossses = all_predictions - new_predictions

                losses.append(local_lossses)

            local_explanation = np.mean(np.array(losses), axis=0)

            local_explanations.append(local_explanation)
            # print(np.std(losses) / np.mean(losses))

            global_explanations.append(np.mean(np.mean(np.abs(local_explanation))))

        self.local_explanations = np.array(local_explanations)

        self.global_explanations = np.array(global_explanations)


class BootstrapInterpreter:
    def __init__(
        self,
        n_iterations: int = 10,
        feature_names: Optional[List[str]] = None,
        downsample_background: bool = False,
        problem_type: str = "classification",
        shuffle_iterations: int = 3,
    ):
        self.percent_cutoff = None
        self.feature_counts = None
        self.n_iterations = n_iterations
        self.feature_names = feature_names
        self.downsample_background = downsample_background
        self.problem_type = problem_type
        self.shuffle_iterations = shuffle_iterations

    def fit(self, X, y, clf) -> None:

        results = dict()

        results["feature"] = self.feature_names

        for i in range(self.n_iterations):
            X_train, y_train = resample(
                X, y, replace=True, n_samples=X.shape[0] * 1, stratify=y, random_state=i
            )

            explainer = FeatureImportance(
                n_iterations=self.shuffle_iterations,
                feature_names=self.feature_names,
                problem_type=self.problem_type,
            )

            if self.downsample_background:
                rus = RandomUnderSampler(random_state=0)

                X_resampled, y_train = rus.fit_resample(X_train, y_train)
                clf.fit(X_resampled, y_train)
                explainer.fit(clf, X_resampled)

            else:
                clf.fit(X_train, y_train)
                explainer.fit(clf, X_train)

            results[f"iteration_{i}_importance"] = pd.Series(
                explainer.global_explanations / explainer.global_explanations.max()
            )
            results[f"iteration_{i}_rank"] = results[f"iteration_{i}_importance"].rank(
                ascending=False
            )

        self.importances = pd.DataFrame(results)

        self.importances["mean_importance"] = self.importances[
            [f"iteration_{i}_importance" for i in range(self.n_iterations)]
        ].mean(axis=1)
        self.importances["median_importance"] = self.importances[
            [f"iteration_{i}_importance" for i in range(self.n_iterations)]
        ].median(axis=1)
        self.importances["stdev_importance"] = self.importances[
            [f"iteration_{i}_importance" for i in range(self.n_iterations)]
        ].std(axis=1)

        self.importances["mean_rank"] = self.importances[
            [f"iteration_{i}_rank" for i in range(self.n_iterations)]
        ].mean(axis=1)
        self.importances["median_rank"] = self.importances[
            [f"iteration_{i}_rank" for i in range(self.n_iterations)]
        ].median(axis=1)
        self.importances["stdev_rank"] = self.importances[
            [f"iteration_{i}_rank" for i in range(self.n_iterations)]
        ].std(axis=1)

    @property
    def results_(self) -> pd.DataFrame:
        return self.importances

    def select_features(
        self,
        top_n: int = 10,
        percent: float = 0.5,
        method: str = "importance",
        metric="percent",
    ) -> List[str]:
        final_features = list()
        all_selected_features = dict()

        if method == "count":
            for i in range(self.n_iterations):
                selected_features = (
                    self.importances.sort_values(
                        f"iteration_{i}_importance", ascending=False
                    )
                    .head(top_n)["feature"]
                    .to_list()
                )

                for feature in selected_features:
                    all_selected_features[feature] = (
                        all_selected_features.get(feature, 0) + 1
                    )

            feature_counts = {
                k: v / self.n_iterations
                for k, v in sorted(
                    all_selected_features.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }

            self.feature_counts = pd.DataFrame(
                {"feature": feature_counts.keys(), "count": feature_counts.values()}
            )

            if metric == "percent":
                self.percent_cutoff = percent

                final_features = self.feature_counts[
                    self.feature_counts["count"] > percent
                ]["feature"].to_list()

            elif metric == "knee":
                sorted_feature_counts = self.feature_counts.sort_values(
                    "count", ascending=False
                ).reset_index(drop=True)

                kn = KneeLocator(
                    sorted_feature_counts.index.values,
                    sorted_feature_counts["count"].values,
                    curve="convex",
                    direction="decreasing",
                )

                self.percent_cutoff = kn.knee_y

                final_features = self.feature_counts[
                    self.feature_counts["count"] > self.percent_cutoff
                ]["feature"].to_list()

        elif method == "importance":
            if metric == "percent":
                self.percent_cutoff = percent

                final_features = self.importances[
                    self.importances["mean_importance"] > self.percent_cutoff
                ]["feature"].to_list()

            elif metric == "knee":
                sorted_feature_counts = self.importances.sort_values(
                    "mean_importance", ascending=False
                ).reset_index(drop=True)

                kn = KneeLocator(
                    sorted_feature_counts.index.values,
                    sorted_feature_counts["mean_importance"].values,
                    curve="convex",
                    direction="decreasing",
                )

                self.percent_cutoff = kn.knee_y

                final_features = self.importances[
                    self.importances["mean_importance"] > self.percent_cutoff
                ]["feature"].to_list()

        return final_features
