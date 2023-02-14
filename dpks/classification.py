from typing import TYPE_CHECKING, Any
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np
import shap


if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class Classifier:
    shap_values: list
    quantitative_data: QuantMatrix

    def __init__(self, classifier, quantitative_data: QuantMatrix):
        self.X, self.Y = self._generate_data_matrices(quantitative_data)
        if isinstance(classifier, str):
            if classifier == "xgboost":
                self.clf = xgboost.XGBClassifier(max_depth=15)
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
        self.clf.fit(self.X, self.Y)
        return self.clf

    def cross_validation(self, k_folds: int = 5):
        scores = cross_val_score(self.clf, self.X, self.Y, cv=k_folds)
        print(
            "%0.2f accuracy with a standard deviation of %0.2f"
            % (scores.mean(), scores.std())
        )

    def interpret(self, algorithm: str = "auto", pass_predict_to_shap: bool = False):
        clf = self.fit()

        if algorithm == "permutation":
            explainer = shap.Explainer(clf.predict, self.X, algorithm=algorithm)
            self.shap_values = explainer(self.X, max_evals=2 * self.X.shape[1] + 1)
        elif algorithm == "tree":
            explainer = shap.Explainer(clf, self.X, algorithm=algorithm)
            self.shap_values = explainer.shap_values(self.X)

    def _generate_data_matrices(self, quantitative_data: QuantMatrix) -> tuple:
        le = LabelEncoder()
        Y = le.fit_transform(quantitative_data.quantitative_data.var["group"].values)
        X = quantitative_data.quantitative_data.X.copy().transpose()
        X = np.nan_to_num(X, copy=True, nan=0.0)
        return X, Y

    @property
    def feature_importance_(self):
        return self.shap_values
