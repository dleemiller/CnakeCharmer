"""Test tridiagonal_solve equivalence."""

import pytest

from cnake_charmer.cy.numerical.tridiagonal_solve import tridiagonal_solve as cy_func
from cnake_charmer.py.numerical.tridiagonal_solve import tridiagonal_solve as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_tridiagonal_solve_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
