"""Test genetic_algorithm equivalence."""

import pytest

from cnake_charmer.cy.optimization.genetic_algorithm import genetic_algorithm as cy_func
from cnake_charmer.py.optimization.genetic_algorithm import genetic_algorithm as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_genetic_algorithm_equivalence(n):
    py = py_func(n)
    cy = cy_func(n)
    for a, b in zip(py, cy):
        assert abs(a - b) / max(abs(a), 1.0) < 1e-6
