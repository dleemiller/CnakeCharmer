"""Test count_paths_grid equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.count_paths_grid import count_paths_grid as cy_func
from cnake_charmer.py.dynamic_programming.count_paths_grid import count_paths_grid as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 300])
def test_count_paths_grid_equivalence(n):
    assert py_func(n) == cy_func(n)
