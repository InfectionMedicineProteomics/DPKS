from typing import Optional

import numpy as np
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
        downsample_rate: float = 0.8
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

    @property
    def ranking_(self):
        return self.selector.ranking_


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
