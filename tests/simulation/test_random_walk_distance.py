"""Test random_walk_distance equivalence."""

import pytest

from cnake_charmer.cy.simulation.random_walk_distance import random_walk_distance as cy_func
from cnake_charmer.py.simulation.random_walk_distance import random_walk_distance as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_random_walk_distance_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
