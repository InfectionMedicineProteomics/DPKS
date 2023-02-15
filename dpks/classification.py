from typing import TYPE_CHECKING, Any
import xgboost
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import shap
from sklearn.feature_selection import RFECV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class Classifier(BaseEstimator, ClassifierMixin):
    X: np.array
    Y: np.array
    shap_values: list
    shap_algorithm: str
    quantitative_data: QuantMatrix
    mean_importance: list

    def __init__(
        self,
        classifier,
        quantitative_data: QuantMatrix,
        scale: bool = True,
        shap_algorithm: str = "auto",
    ):
        self.X, self.Y = self._generate_data_matrices(quantitative_data, scale)
        if isinstance(classifier, str):
            if classifier == "xgboost":
                self.classifier = xgboost.XGBClassifier(max_depth=30)
        else:
            fit_method = getattr(classifier, "fit", None)
            predict_method = getattr(classifier, "predict", None)
            if callable(fit_method) and callable(predict_method):
                self.classifier = classifier
            else:
                raise ValueError(
                    "The classifier does not have a fit and/or predict method"
                )
        self.quantitative_data = quantitative_data
        self.scale = scale
        self.shap_algorithm = shap_algorithm

    def fit(self, X, Y):
        self.classifier.fit(X, Y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def cross_validation(self, k_folds: int = 5):
        scores = cross_val_score(self.classifier, self.X, self.Y, cv=k_folds)
        print(
            "%0.2f accuracy with a standard deviation of %0.2f"
            % (scores.mean(), scores.std())
        )

    def recursive_feature_elimination(
        self,
        k_folds: int = 3,
        scoring: str = "accuracy",
        min_features_to_select: int = 10,
        step: int = 3,
        importance_getter: str = "auto",
    ):
        selector = RFECV(
            estimator=self,
            step=step,
            min_features_to_select=min_features_to_select,
            scoring=scoring,
            cv=k_folds,
            importance_getter=importance_getter,
        )

        selector = selector.fit(self.X, self.Y)
        print(f"Number of selected features: {selector.n_features_}")
        self.quantitative_data.row_annotations[f"RFE_ranking"] = selector.ranking_
        return selector.cv_results_

    def interpret(self, annotate: bool = True) -> QuantMatrix:
        self.fit(
            self.X, self.Y
        )  # TODO: I guess this is wrong since self.X is wrong size.

        if self.shap_algorithm == "permutation":
            explainer = shap.Explainer(
                self.classifier.predict, self.X, algorithm=self.shap_algorithm
            )
            self.shap_values = explainer(self.X, max_evals=2 * self.X.shape[1] + 1)
        elif self.shap_algorithm == "tree" or self.shap_algorithm == "auto":
            explainer = shap.Explainer(self.classifier, algorithm=self.shap_algorithm)
            self.shap_values = explainer.shap_values(self.X)

        self.mean_importance = np.mean(abs(self.shap_values), axis=0)

        if annotate:
            self.quantitative_data.row_annotations[f"SHAP"] = self.mean_importance

        return self.quantitative_data

    def _generate_data_matrices(
        self, quantitative_data: QuantMatrix, scale: bool
    ) -> tuple:
        le = LabelEncoder()
        Y = le.fit_transform(quantitative_data.quantitative_data.var["group"].values)
        X = quantitative_data.quantitative_data.X.copy().transpose()
        X = np.nan_to_num(X, copy=True, nan=0.0)
        if scale:
            X = StandardScaler().fit_transform(X)
        return X, Y

    @property
    def feature_importances_(self):
        self.interpret(annotate=False)
        return self.mean_importance
