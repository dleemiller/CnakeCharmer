"""Test min_cost_path equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.min_cost_path import min_cost_path as cy_func
from cnake_data.py.dynamic_programming.min_cost_path import min_cost_path as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_min_cost_path_equivalence(n):
    assert py_func(n) == cy_func(n)
