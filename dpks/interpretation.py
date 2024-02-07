from typing import Optional, List

import pandas as pd

from sklearn.utils import resample

from dpks.classification import Classifier

from imblearn.under_sampling import RandomUnderSampler
from kneed import KneeLocator

class BootstrapInterpreter:
    def __init__(
        self,
        n_iterations: int = 10,
        feature_names: Optional[List[str]] = None,
        downsample_background: bool = False
    ):
        self.percent_cutoff = None
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

        final_features = list()
        all_selected_features = dict()

        if method == "count":

            for i in range(self.n_iterations):

                selected_features = (
                    self.importances.sort_values(
                        f"iteration_{i}_shap", ascending=False
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
                    all_selected_features.items(), key=lambda item: item[1], reverse=True
                )
            }

            self.feature_counts = pd.DataFrame(
                {
                    "feature": feature_counts.keys(),
                    "count": feature_counts.values()
                }
            )

            if metric == "percent":

                self.percent_cutoff = percent

                final_features = self.feature_counts[self.feature_counts['count'] > percent]['feature'].to_list()

            elif metric == "knee":

                sorted_feature_counts = self.feature_counts.sort_values("count", ascending=False).reset_index(drop=True)

                kn = KneeLocator(
                    sorted_feature_counts.index.values,
                    sorted_feature_counts['count'].values,
                    curve='convex',
                    direction='decreasing'
                )

                self.percent_cutoff = kn.knee_y

                final_features = self.feature_counts[self.feature_counts['count'] > self.percent_cutoff]['feature'].to_list()

        elif method == "shap":

            # self.importances['shap_scaled'] = self.importances['mean_shap'] / self.importances['mean_shap'].sum()

            if metric == "percent":

                self.percent_cutoff = percent

                final_features = self.importances[self.importances['mean_shap'] > self.percent_cutoff]['feature'].to_list()

            elif metric == "knee":

                sorted_feature_counts = self.importances.sort_values("mean_shap", ascending=False).reset_index(drop=True)

                kn = KneeLocator(
                    sorted_feature_counts.index.values,
                    sorted_feature_counts['mean_shap'].values,
                    curve='convex',
                    direction='decreasing'
                )

                self.percent_cutoff = kn.knee_y

                final_features = self.importances[
                    self.importances['mean_shap'] > self.percent_cutoff
                ]['feature'].to_list()

        return final_features
