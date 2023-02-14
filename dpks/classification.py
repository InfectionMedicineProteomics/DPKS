from typing import TYPE_CHECKING, Any
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


if TYPE_CHECKING:
    from .quant_matrix import QuantMatrix
else:
    QuantMatrix = Any


class Classifier:
    def __init__(
        self,
        classifier,
    ):
        if isinstance(classifier, str):
            if classifier == "xgboost":
                self.clf = xgboost.XGBClassifier()
        else:
            fit_method = getattr(classifier, "fit", None)
            predict_method = getattr(classifier, "predict", None)
            if callable(fit_method) and callable(predict_method):
                self.clf = classifier
            else:
                raise ValueError(
                    "The classifier does not have a fit and/or predict method"
                )

    def fit(self, quantitative_data: QuantMatrix):
        X, Y = self._generate_data_matrices(quantitative_data)
        print(X.shape, Y.shape)
        self.clf.fit(X, Y)

    def cross_validation(self, quantitative_data: QuantMatrix, k_folds: int = 5):
        X, Y = self._generate_data_matrices(quantitative_data)
        scores = cross_val_score(self.clf, X, Y, cv=k_folds)
        print(
            "%0.2f accuracy with a standard deviation of %0.2f"
            % (scores.mean(), scores.std())
        )

    def _generate_data_matrices(self, quantitative_data: QuantMatrix) -> tuple:
        le = LabelEncoder()
        Y = le.fit_transform(quantitative_data.quantitative_data.var["group"].values)
        X = quantitative_data.quantitative_data.X.copy().transpose()
        return X, Y
