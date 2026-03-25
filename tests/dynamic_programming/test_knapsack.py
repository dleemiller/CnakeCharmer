"""Test knapsack equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.knapsack import knapsack as cy_knapsack
from cnake_charmer.py.dynamic_programming.knapsack import knapsack as py_knapsack


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_knapsack_equivalence(n):
    assert py_knapsack(n) == cy_knapsack(n)
