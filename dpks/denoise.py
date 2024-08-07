import numpy as np
import numpy.typing as npt

from typing import Optional, Any

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold


class BaggedDenoiser(BaggingClassifier):  # type: ignore
    def __init__(
        self,
        estimator: Optional[Any] = None,
        n_estimators: int = 250,
        max_samples: float = 1.0,
        n_jobs: int = 5,
        random_state: int = 0,
    ):

        if not estimator:

            estimator = SGDClassifier(
                alpha=1e-05,
                average=True,
                loss="log_loss",
                max_iter=500,
                penalty="l2",
                shuffle=True,
                tol=0.0001,
                learning_rate="adaptive",
                eta0=0.001,
                fit_intercept=True,
                random_state=random_state
            )

        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def vote(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> np.ndarray:

        skf = StratifiedKFold(n_splits=3)

        vote_percentages = np.zeros((X.shape[0],))

        for i, (train, test) in enumerate(skf.split(X, y)):

            X_fold = X[train]
            y_fold = y[train]

            self.fit(X_fold, y_fold)

            estimator_probabilities = list()

            for estimator in self.estimators_:

                probabilities = np.where(
                    estimator.predict_proba(X_fold)[:, 1] >= threshold, 1, 0
                )

                estimator_probabilities.append(probabilities)

            estimator_probability_array = np.array(
                estimator_probabilities, dtype=np.float64
            )

            vote_percentages[train] = estimator_probability_array.sum(axis=0) / len(self.estimators_)

        return vote_percentages
    
    def predict_proba(self, X) -> np.ndarray:

        votes = self.vote(X)

        probabilities = np.zeros(
            (votes.shape[0], 2)
        )

        probabilities[:, 0] = 1 - votes
        probabilities[:, 1] = votes

        return probabilities

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return np.where(
            probabilities[:, 1] > 0.5, 1, 0
        )
