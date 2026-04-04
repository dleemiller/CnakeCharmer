"""Test random_walk_2d equivalence."""

import pytest

from cnake_charmer.cy.simulation.random_walk_2d import random_walk_2d as cy_random_walk_2d
from cnake_charmer.py.simulation.random_walk_2d import random_walk_2d as py_random_walk_2d


@pytest.mark.parametrize("n", [1000, 50000, 500000, 5000000])
def test_random_walk_2d_equivalence(n):
    py_result = py_random_walk_2d(n)
    cy_result = cy_random_walk_2d(n)
    assert py_result == cy_result, f"Mismatch at n={n}: py={py_result}, cy={cy_result}"
