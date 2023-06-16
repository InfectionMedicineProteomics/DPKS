from typing import Any

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
import random
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler


class ParamSearch:

    def __init__(self) -> None:

        pass

    def fit(self, classifier, X, y, **kwargs):

        pass


class ParamSearchResult:

    def __init__(self, classifier, result: Any) -> None:

        self.classifier = classifier
        self.result = result

class RandomizedSearch:

    def __init__(
        self,
        classifier,
        param_grid: dict,
        folds: int = 3,
        random_state: int = None,
        n_iter: int = 30,
        n_jobs: int = 4,
        scoring: str = "accuracy",
        verbose: bool = False
    ):

        self.classifier = classifier
        self.param_grid = param_grid
        self.folds = folds
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.verbose = 4 if verbose else 0

    def fit(self, X, y) -> ParamSearchResult:

        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)

        random_search = RandomizedSearchCV(
            self.classifier,
            param_distributions=self.param_grid,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
            cv=skf.split(X, y),
            verbose=self.verbose,
            scoring=self.scoring,
            return_train_score=True
        )

        random_search.fit(X, y)

        return ParamSearchResult(
            classifier=random_search.best_estimator_,
            result=random_search
        )



class GeneticAlgorithmSearch(ParamSearch):
    def __init__(
        self,
        classifier,
        param_grid: dict,
        n_generations: int = 50,
        pop_size: int = 10,
        n_survive: int = 5,
        threads: int = 1,
        folds: int = 3,
        verbose: bool = False,
    ) -> None:
        self.classifier = classifier
        self.param_grid = param_grid
        self.threads = threads
        self.pop_size = pop_size
        self.folds = folds
        self.n_generations = n_generations
        self.n_survive = n_survive
        self.n_procreate = pop_size - n_survive
        self.populations = {}
        self.verbose = verbose

    def get_accuracy(self, classifier, X, y):
        scores = cross_val_score(classifier, X, y, n_jobs=self.threads, cv=self.folds)
        return np.mean(scores)

    def initiate_pop(self, param_grid):
        pop = []
        for _ in range(self.pop_size):
            random_initializer = random.uniform(0, 1)
            individual = {}
            for param in param_grid.keys():
                individual[param] = random.choice(param_grid[param])
            pop.append((random_initializer, individual))
        return pop

    def generation_pass(self, pop, X, y):
        evaluated_pop = []
        for _, params in pop:
            clf_individual = clone(self.classifier)
            clf_individual.set_params(**params)
            acc = self.get_accuracy(clf_individual, X, y)
            evaluated_pop.append((acc, params))
        return evaluated_pop

    def kill(self, pop):
        pop.sort(key=lambda x: x[0], reverse=True)
        reduced_pop = pop[: self.n_survive]
        return reduced_pop

    def procreate(self, pop):
        new_pop = pop.copy()
        for _ in range(self.n_procreate):
            individual = {}
            random_initializer = random.uniform(0, 1)
            for oldie in pop:
                _, old_individual = oldie
                for param in old_individual.keys():
                    individual[param] = random.choice(self.param_grid[param])
            new_pop.append((random_initializer, individual))
        return new_pop

    def fit(self, X, y) -> dict:
        pop = self.initiate_pop(self.param_grid)
        for generation in range(self.n_generations):
            evaluated_pop = self.generation_pass(pop, X, y)
            reduced_pop = self.kill(evaluated_pop)
            if self.verbose:
                print(f"Generation {generation}")
                print(f"Accuracy {reduced_pop[0][0]}")
                print(f"Best param {reduced_pop[0][1]}")
            self.populations[generation] = reduced_pop
            pop = self.procreate(reduced_pop)
        return self.populations

    @property
    def best_estimator_(self):
        if len(self.populations) == 0:
            raise Exception("The genetic algorithm has not been run")
        acc, best_params = self.populations[0][0]
        if self.verbose:
            print("Accuracy: ", acc)
        best_estimator = clone(self.classifier)
        best_estimator.set_params(**best_params)
        return best_estimator
