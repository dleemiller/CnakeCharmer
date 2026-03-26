"""Test simplex_nelder_mead equivalence."""

import pytest

from cnake_charmer.cy.optimization.simplex_nelder_mead import simplex_nelder_mead as cy_func
from cnake_charmer.py.optimization.simplex_nelder_mead import simplex_nelder_mead as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_simplex_nelder_mead_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
