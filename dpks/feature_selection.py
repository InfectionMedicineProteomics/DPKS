from typing import Optional, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import RFE
from sklearn.model_selection import (
    cross_val_score,
)
from sklearn.utils import resample


class BoostrapRFE:
    """
    Bootstrap Recursive Feature Elimination (RFE) for feature selection.

    Parameters:
    - min_features_to_select (int): Minimum number of features to select.
    - step (int): Step size for RFE.
    - importance_getter (str): Method to get feature importances.
    - scoring (str): Scoring metric for feature selection.
    - k_folds (int): Number of folds for cross-validation.
    - threads (int): Number of threads for parallel processing.
    - verbose (bool): Verbosity of the process.
    - shap_algorithm (str): SHAP algorithm to use.
    - random_state (int): Random seed for reproducibility.
    - shuffle (bool): Whether to shuffle data during resampling.
    - replace (bool): Whether to use replacement during resampling.
    - downsample_rate (float): Rate of downsampling during resampling.
    - feature_names (Optional[List[str]]): List of feature names.

    Methods:
    - fit(X, y, classifier): Fit the BootstrapRFE instance to the data.
    - get_scores(): Get the scores of feature selectors.
    - get_ranks(): Get the rankings of selected features.
     - evaluate(X, y, classifier): Evaluate the performance with feature subsets.

    """

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
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the BoostrapRFE instance with specified parameters.

        Parameters:
        ...

        Returns:
        None
        """
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

    def fit(self, X, y, classifier) -> None:
        """
        Fit the BootstrapRFE instance to the data.

        Parameters:
        - X: Input data.
        - y: Target values.
        - classifier: Classifier for feature selection.

        Returns:
        None
        """

        self.results = Parallel(n_jobs=self.threads)(
            delayed(_rank_features)(
                classifier,
                X,
                y,
                self.scoring,
                self.replace,
                self.downsample_rate,
                self.step,
                self.min_features_to_select,
                self.importance_getter,
            )
            for i in range(self.k_folds)
        )

    def get_scores(self):
        """
        Get the scores of feature selectors.

        Returns:
        pd.DataFrame: DataFrame containing feature scores.
        """

        selector_scores = []
        feature_number = []

        for idx, (_, scores) in enumerate(self.results):
            for score in scores:
                selector_scores.append(scores[score])
                feature_number.append(score)

        scores = pd.DataFrame(
            {"feature_number": feature_number, "score": selector_scores}
        )

        return scores

    def get_ranks(self):
        """
        Get the rankings of selected features.

        Returns:
        pd.DataFrame: DataFrame containing feature rankings.
        """

        selector_rankings = dict()

        for idx, (selector, _) in enumerate(self.results):
            selector_rankings[idx] = selector.ranking_

        feature_ranks = pd.DataFrame(
            {
                "feature_name": self.feature_names,
            }
        )

        for idx, ranking in selector_rankings.items():
            feature_ranks[f"ranking_{idx}"] = ranking

        feature_ranks = feature_ranks.copy()

        feature_ranks["feature_rank_mean"] = feature_ranks[
            [col for col in feature_ranks.columns if "ranking_" in col]
        ].mean(axis=1)
        feature_ranks["feature_rank_std"] = feature_ranks[
            [col for col in feature_ranks.columns if "ranking_" in col]
        ].std(axis=1)
        feature_ranks["feature_rank_median"] = feature_ranks[
            [col for col in feature_ranks.columns if "ranking_" in col]
        ].median(axis=1)

        feature_ranks["adjusted_rank"] = feature_ranks["feature_rank_median"] * np.log(
            feature_ranks["feature_rank_std"]
        )

        return feature_ranks

    def evaluate(self,  X, y, classifier):

        """
               Evaluate the performance with feature subsets.

               Parameters:
               - X: Input data.
               - y: Target values.
               - classifier: Classifier for feature selection.

               Returns:
               pd.DataFrame: DataFrame containing evaluation results.
               """

        rank_values = []
        rank_scores = []

        feature_ranks = self.get_ranks()

        for rank_value in feature_ranks.sort_values("adjusted_rank", ascending=False)['adjusted_rank'].values:

            if isinstance(X, pd.DataFrame):

                selected_proteins = feature_ranks[feature_ranks['adjusted_rank'] <= rank_value]['feature_name'].values

                X_subset = X[selected_proteins]

            ## Need numpy array subsetting here also

            scores = cross_val_score(
                classifier,
                X_subset,
                y,
                scoring='accuracy',
                cv=3
            )

            for score in scores:
                rank_scores.append(score)
                rank_values.append(rank_value)

        return pd.DataFrame(
            {
                "rank": rank_values,
                "score": rank_scores
            }
        )


def _rank_features(
    clf,
    X,
    y,
    scoring,
    replace,
    downsample_rate,
    rfe_step,
    rfe_min_features,
    rfe_importance_getter,
) -> tuple[RFE, dict]:
    """
    Rank features using Recursive Feature Elimination (RFE).

    Parameters:
    - clf: Classifier for feature selection.
    - X: Input data.
    - y: Target values.
    - scoring: Scoring metric for feature selection.
    - replace: Whether to use replacement during resampling.
    - downsample_rate: Rate of downsampling during resampling.
    - rfe_step: Step size for RFE.
    - rfe_min_features: Minimum number of features to select.
    - rfe_importance_getter: Method to get feature importances.

    Returns:
    Tuple: Tuple containing the feature selector and a dictionary of feature scores.
    """
    scores = dict()

    X_train, y_train = resample(
        X, y, replace=replace, n_samples=X.shape[0] * downsample_rate, stratify=y
    )

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
        if isinstance(X_train, pd.DataFrame):
            X_subset = X_train.loc[:, (selector.ranking_ <= feature_num)]

        else:
            X_subset = X_train[:, (selector.ranking_ <= feature_num)]

        if X_subset.ndim < 2:
            X_subset = X_subset.reshape(-1, 1)

        score = cross_val_score(clf, X_subset, y_train, scoring=scoring, cv=3, n_jobs=1)

        scores[feature_num] = np.mean(score)

    return selector, scores
