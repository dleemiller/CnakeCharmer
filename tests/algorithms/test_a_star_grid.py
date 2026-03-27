"""Test a_star_grid equivalence."""

import pytest

from cnake_charmer.cy.algorithms.a_star_grid import a_star_grid as cy_func
from cnake_charmer.py.algorithms.a_star_grid import a_star_grid as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_a_star_grid_equivalence(n):
    assert py_func(n) == cy_func(n)
