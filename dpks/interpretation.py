from typing import Optional, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import RFE
from sklearn.model_selection import (
    cross_val_score,
)
from sklearn.utils import resample

from dpks.classification import Classifier

from imblearn.under_sampling import RandomUnderSampler


class BootstrapInterpreter:
    def __init__(
        self,
        n_iterations: int = 10,
        feature_names: Optional[List[str]] = None,
        downsample_background: bool = False
    ):
        self.feature_counts = None
        self.n_iterations = n_iterations
        self.feature_names = feature_names
        self.downsample_background = downsample_background

    def fit(self, X, y, classifier) -> None:
        results = dict()

        results["feature"] = self.feature_names

        for i in range(self.n_iterations):
            X_train, y_train = resample(
                X, y, replace=True, n_samples=X.shape[0] * 1, stratify=y, random_state=i
            )

            if isinstance(classifier, Classifier):
                clf = classifier
            else:
                clf = Classifier(classifier=classifier)

            clf.fit(X_train, y_train)

            if self.downsample_background:
                rus = RandomUnderSampler(random_state=0)
                X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
                clf.interpret(X_resampled)
            else:
                clf.interpret(X_train)

            results[f"iteration_{i}_shap"] = pd.Series(
                clf.mean_importance / clf.mean_importance.max()
            )
            results[f"iteration_{i}_rank"] = results[f"iteration_{i}_shap"].rank(
                ascending=False
            )

        self.importances = pd.DataFrame(results)

        self.importances["mean_shap"] = self.importances[
            [f"iteration_{i}_shap" for i in range(self.n_iterations)]
        ].mean(axis=1)
        self.importances["median_shap"] = self.importances[
            [f"iteration_{i}_shap" for i in range(self.n_iterations)]
        ].median(axis=1)
        self.importances["stdev_shap"] = self.importances[
            [f"iteration_{i}_shap" for i in range(self.n_iterations)]
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
        method: str = "shap",
        metric="percent",
    ) -> List[str]:
        all_selected_features = dict()

        for i in range(self.n_iterations):
            if method == "shap":
                selected_features = (
                    self.importances.sort_values(
                        f"iteration_{i}_{method}", ascending=False
                    )
                    .head(top_n)["feature"]
                    .to_list()
                )

            elif method == "rank":
                selected_features = (
                    self.importances.sort_values(
                        f"iteration_{i}_{method}", ascending=True
                    )
                    .head(top_n)["feature"]
                    .to_list()
                )

            for feature in selected_features:
                all_selected_features[feature] = (
                    all_selected_features.get(feature, 0) + 1
                )

        self.feature_counts = {
            k: v / self.n_iterations
            for k, v in sorted(
                all_selected_features.items(), key=lambda item: item[1], reverse=True
            )
        }

        return [
            feature for feature, count in self.feature_counts.items() if count > percent
        ]
