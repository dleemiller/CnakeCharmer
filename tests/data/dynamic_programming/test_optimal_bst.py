"""Test optimal_bst equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.optimal_bst import optimal_bst as cy_func
from cnake_data.py.dynamic_programming.optimal_bst import optimal_bst as py_func


@pytest.mark.parametrize("n", [5, 10, 50, 100])
def test_optimal_bst_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
