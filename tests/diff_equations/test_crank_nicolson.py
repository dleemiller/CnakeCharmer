"""Test crank_nicolson equivalence."""

import pytest

from cnake_charmer.cy.diff_equations.crank_nicolson import crank_nicolson as cy_func
from cnake_charmer.py.diff_equations.crank_nicolson import crank_nicolson as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_crank_nicolson_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
