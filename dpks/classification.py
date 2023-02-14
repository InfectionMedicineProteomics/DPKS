from typing import TYPE_CHECKING, Any
import xgboost
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import shap
from sklearn.feature_selection import RFECV


if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class Classifier:
    X: np.array
    Y: np.array
    shap_values: list
    mean_importance: list

    def __init__(self, classifier, quantitative_data: QuantMatrix, scale: bool = True):
        self.X, self.Y = self._generate_data_matrices(quantitative_data, scale)
        if isinstance(classifier, str):
            if classifier == "xgboost":
                self.clf = xgboost.XGBClassifier(max_depth=30)
        else:
            fit_method = getattr(classifier, "fit", None)
            predict_method = getattr(classifier, "predict", None)
            if callable(fit_method) and callable(predict_method):
                self.clf = classifier
            else:
                raise ValueError(
                    "The classifier does not have a fit and/or predict method"
                )

    def fit(self):
        return self.clf.fit(self.X, self.Y)

    def cross_validation(self, k_folds: int = 5):
        scores = cross_val_score(self.clf, self.X, self.Y, cv=k_folds)
        print(
            "%0.2f accuracy with a standard deviation of %0.2f"
            % (scores.mean(), scores.std())
        )

    def recursive_feature_elimination(
        self,
        k_folds: int = 5,
        scoring: str = "accuracy",
        min_features_to_select: int = 10,
        step: int = 10,
    ):
        selector = RFECV(
            self.clf,
            step=step,
            min_features_to_select=min_features_to_select,
            scoring=scoring,
            cv=k_folds,
        )

        selector = selector.fit(self.X, self.Y)
        print(f"Number of selected features: {selector.n_features_}")
        print(selector.support_)
        print(selector.ranking_)

    def interpret(self, quantitative_data, algorithm: str = "auto") -> QuantMatrix:
        clf = self.fit()

        if algorithm == "permutation":
            explainer = shap.Explainer(clf.predict, self.X, algorithm=algorithm)
            self.shap_values = explainer(self.X, max_evals=2 * self.X.shape[1] + 1)
        elif algorithm == "tree":
            explainer = shap.Explainer(clf, algorithm=algorithm)
            self.shap_values = explainer.shap_values(self.X)
        elif algorithm == "auto":
            explainer = shap.Explainer(clf, self.X, algorithm=algorithm)
            self.shap_values = explainer.shap_values(self.X)

        self.mean_importance = np.mean(abs(self.shap_values), axis=0)
        quantitative_data.row_annotations[f"SHAP"] = self.mean_importance

        return quantitative_data

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
        return self.mean_importance
