import numpy as np
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    StratifiedKFold,
)

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
        shuffle: bool = False
    ) -> None:
        self.selector = None
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
        classifier,
    ) -> None:
        estimator = Classifier(classifier=classifier)
        estimator.shap_algorithm = self.shap_algorithm

        selector = RFE(
            estimator=estimator,
            step=self.step,
            n_features_to_select=1,
            importance_getter=self.importance_getter,
        )

        if self.verbose:
            print(f"Fitting initial selector.")

        selector.fit(X, y)

        for feature_num in range(np.unique(selector.ranking_)):
            X_subset = X[:, (selector.ranking_ <= feature_num)]

            if self.verbose:
                print(f"Evaluating features below rank: {feature_num}")

            skf = StratifiedKFold(n_splits=self.k_folds, shuffle=self.shuffle, random_state=self.random_state)

            scores = list()

            for i, (train_idx, test_idx) in enumerate(skf.split(X_subset, y)):
                X_train = X_subset[train_idx, :]
                y_train = y[train_idx]

                X_test = X_subset[test_idx, :]
                y_test = y[test_idx]

                if X_train.ndim < 2:
                    X_train = X_train.reshape(-1, 1)
                    X_test = X_test.reshape(-1, 1)

                clf = Classifier(classifier=classifier)

                clf.fit(X_train, y_train)

                score = accuracy_score(y_test, clf.predict(X_test))

                scores.append(score)

            self.results[feature_num] = scores

            if self.verbose:
                print(
                    f"Model ({feature_num} features): {np.mean(scores)} {np.std(scores)}"
                )

        self.selector = selector

    @property
    def ranking_(self):
        return self.selector.ranking_
