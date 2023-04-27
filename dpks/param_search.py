import numpy as np
from sklearn.model_selection import cross_val_score
import random
from sklearn.base import clone


class GeneticAlgorithmSearch:
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

    def run_genetic_algorithm(self, X, y) -> dict:
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
