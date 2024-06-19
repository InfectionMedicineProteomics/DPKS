import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class FeatureImportance:

    all_predictions: np.ndarray
    feature_names: list[str]
    def __init__(self, n_iterations: int = 3, feature_names: list[str] = None):
        self.n_iterations = n_iterations
        self.feature_names = feature_names

    def fit(self, clf, X) -> None:

        decision_function = getattr(clf, "decision_function", None)

        if callable(decision_function):
            all_predictions = clf.decision_function(X)
        else:
            all_predictions = clf.predict_proba(X)[:, 1]

        global_explanations = []

        local_explanations = []

        for i in range(X.shape[1]):

            feature_slice = X[:, i].copy()

            X_copy = X.copy()

            losses = []

            for _ in range(self.n_iterations):
                np.random.shuffle(feature_slice)

                X_copy[:, i] = feature_slice

                if callable(decision_function):
                    new_predictions = clf.decision_function(X_copy)
                else:
                    new_predictions = clf.predict_proba(X_copy)[:, 1]

                local_lossses = all_predictions - new_predictions

                losses.append(
                    local_lossses
                )

            local_explanation = np.mean(np.array(losses), axis=0)

            local_explanations.append(local_explanation)
            # print(np.std(losses) / np.mean(losses))

            global_explanations.append(
                np.mean(
                    np.mean(np.abs(local_explanation))
                )
            )

        self.local_explanations = np.array(local_explanations)

        self.global_explanations = np.array(global_explanations)


