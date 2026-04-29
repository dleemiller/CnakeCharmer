"""Tournament-based function selector with fitness tracking."""

from __future__ import annotations

import random


class FunctionSelection:
    def __init__(self, nfunctions=0, seed=0, tournament_size=2, nargs=None, density_safe=None):
        self.fitness = [0.0] * nfunctions
        self.times = [0] * nfunctions
        self.nargs = [0] * nfunctions
        self.unfeasible_functions = set()
        if nargs is not None:
            for i, k in enumerate(nargs):
                self.nargs[i] = k
        self.density_safe = list(density_safe) if density_safe is not None else []
        self.density_safe_size = len(self.density_safe)
        self.nfunctions = nfunctions
        self.tournament_size = tournament_size
        self.min_density = 0.0
        self.density = 1.0
        random.seed(seed)

    def update(self, k, v):
        self.fitness[k] += v
        self.times[k] += 1

    def avg_fitness(self, idx):
        return 0.0 if self.times[idx] == 0 else self.fitness[idx] / self.times[idx]

    def random_function(self):
        pool = (
            self.density_safe
            if self.density < self.min_density and self.density_safe_size > 0
            else list(range(self.nfunctions))
        )
        r = 0
        for _ in range(5):
            r = random.choice(pool)
            if r not in self.unfeasible_functions:
                return r
        return r

    def tournament(self):
        if self.nfunctions == 1:
            return 0
        best = self.random_function()
        best_fit = self.avg_fitness(best)
        for _ in range(1, self.tournament_size):
            comp = self.random_function()
            while (
                comp == best
                and comp not in self.unfeasible_functions
                and len(self.unfeasible_functions) < (self.nfunctions - 1)
            ):
                comp = self.random_function()
            comp_fit = self.avg_fitness(comp)
            if comp_fit > best_fit or (
                comp_fit == best_fit and self.nargs[comp] > self.nargs[best]
            ):
                best_fit = comp_fit
                best = comp
        return best
