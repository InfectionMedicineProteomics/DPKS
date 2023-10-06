from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.utils import resample

from joblib import Parallel, delayed

class FeatureRankerRFE:
    def __init__(
        self,
        min_features_to_select: int = 10,
        step: int = 3,
        importance_getter: str = "auto",
        scoring: str = "accuracy",
        k_folds: int = 3,
        threads: int = 1,
        verbose: bool = False,
        shap_algorithm: str = "auto",
        random_state: int = 42,
        shuffle: bool = True,
        replace: bool = True,
        downsample_rate: float = 0.8,
        feature_names: Optional[List[str]] = None
    ) -> None:
        self.selectors = dict()
        self.results = dict()
        self.verbose = verbose
        self.threads = threads
        self.scoring = scoring
        self.k_folds = k_folds
        self.models = dict()
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.importance_getter = importance_getter
        self.shap_algorithm = shap_algorithm
        self.random_state = random_state
        self.shuffle = shuffle
        self.replace = replace
        self.downsample_rate = downsample_rate
        self.feature_names = feature_names

    def fit(
        self,
        X,
        y,
        classifier
    ) -> None:

        self.results = Parallel(n_jobs=self.threads)(
            delayed(_rank_features)(
                classifier, X, y, self.scoring, self.replace, self.downsample_rate, self.step, self.min_features_to_select, self.importance_getter
            ) for i in range(self.k_folds)
        )

    def get_scores(self):

        selector_scores = []
        feature_number = []

        for idx, (_, scores) in enumerate(self.results):

            for score in scores:
                selector_scores.append(scores[score])
                feature_number.append(score)

        scores = pd.DataFrame(
            {
                "feature_number": feature_number,
                "score": selector_scores
            }
        )

        return scores

    def get_ranks(self):

        selector_rankings = dict()

        for idx, (selector, _) in enumerate(self.results):

            selector_rankings[idx] = selector.ranking_

        feature_ranks = pd.DataFrame(
            {
                "feature_name": self.feature_names,
            }
        )

        for idx, ranking in selector_rankings.items():

            feature_ranks[f'ranking_{idx}'] = ranking

        feature_ranks = feature_ranks.copy()

        feature_ranks['feature_rank_mean'] = feature_ranks[
            [col for col in feature_ranks.columns if "ranking_" in col]].mean(axis=1)
        feature_ranks['feature_rank_std'] = feature_ranks[
            [col for col in feature_ranks.columns if "ranking_" in col]].std(axis=1)
        feature_ranks['feature_rank_median'] = feature_ranks[
            [col for col in feature_ranks.columns if "ranking_" in col]].median(axis=1)

        feature_ranks['adjusted_rank'] = feature_ranks['feature_rank_median'] * np.log(feature_ranks['feature_rank_std'])

        return feature_ranks


def _rank_features(clf, X, y, scoring, replace, downsample_rate, rfe_step, rfe_min_features, rfe_importance_getter):

    scores = dict()

    X_train, y_train = resample(X, y, replace=replace, n_samples=X.shape[0] * downsample_rate, stratify=y)

    selector = RFE(
        estimator=clf,
        step=rfe_step,
        n_features_to_select=rfe_min_features,
        importance_getter=rfe_importance_getter,
    )

    selector.fit(X_train, y_train)

    baseline_score = selector.score(X_train, y_train)

    print(f"Fit initial selector with score {baseline_score}.")

    for feature_num in np.unique(selector.ranking_):

        X_subset = X_train[:, (selector.ranking_ <= feature_num)]

        if X_subset.ndim < 2:
            X_subset = X_subset.reshape(-1, 1)

        score = cross_val_score(clf, X_subset, y_train, scoring=scoring, cv=3, n_jobs=1)

        scores[feature_num] = np.mean(score)

    return selector, scores
