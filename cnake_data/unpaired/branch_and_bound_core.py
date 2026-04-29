"""Core branch-and-bound search loop with pluggable node strategy."""

from __future__ import annotations

import math


class BnBNode:
    def __init__(self):
        self.lowerb = -math.inf
        self.upperb = math.inf
        self.objective_value = math.inf
        self.solution = None
        self.var_to_branch = -1


class BranchAndBound:
    def __init__(self, selection_method, eps=1e-6):
        self.root = BnBNode()
        self.selection_method = selection_method
        self.eps = eps
        self.optimal_solution = None
        self.optimal_objective_value = math.inf

    def run(self):
        active_old = self.root
        while True:
            active_new = self.selection_method.get_active_node(active_old)
            if active_new is None:
                break

            if active_new.solution is not None:
                if self.optimal_objective_value > self.root.objective_value:
                    self.optimal_solution = list(self.root.solution)
                    self.optimal_objective_value = self.root.objective_value

                close = (
                    active_new.lowerb >= self.root.upperb
                    or abs(active_new.lowerb - active_new.upperb) < self.eps
                    or active_new.var_to_branch == -1
                )
                if not close:
                    self.selection_method.create_nodes(active_new.var_to_branch, active_new)

            active_old = active_new

        return self.optimal_solution, self.optimal_objective_value
