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

from dpks.classification import Classifier


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
        shuffle: bool = True
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

    def _evaluate_model(self, classifier, X, y):
        cv = RepeatedStratifiedKFold(n_splits=self.k_folds, random_state=42)

        scores = cross_val_score(
            classifier,
            X,
            y,
            scoring=self.scoring,
            cv=cv,
            n_jobs=self.threads,
        )

        return scores

    def rank_features(
        self,
        X,
        y,
        classifier
    ) -> None:

        for i in range(self.k_folds):

            self.results[i] = dict()

            X_train, y_train = resample(X, y, replace=True, n_samples=X.shape[0] * 0.8, random_state=42, stratify=y)

            if X_train.ndim < 2:
                X_train = X_train.reshape(-1, 1)

            selector = RFE(
                estimator=classifier,
                step=self.step,
                n_features_to_select=1,
                importance_getter=self.importance_getter,
            )

            if self.verbose:
                print(f"Fitting initial selector.")

            selector.fit(X_train, y_train)

            for feature_num in np.unique(selector.ranking_):

                X_subset = X_train[:, (selector.ranking_ <= feature_num)]

                if X_subset.ndim < 2:
                    X_subset = X_subset.reshape(-1, 1)

                score = cross_val_score(classifier, X_subset, y_train, scoring="accuracy", cv=3)

                self.results[i][feature_num] = score

            self.selectors[i] = selector

    @property
    def ranking_(self):
        return self.selector.ranking_
