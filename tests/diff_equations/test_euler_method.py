"""Test euler_method equivalence."""

import pytest

from cnake_charmer.cy.diff_equations.euler_method import euler_method as cy_func
from cnake_charmer.py.diff_equations.euler_method import euler_method as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_euler_method_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-9, f"Mismatch: py={py_result}, cy={cy_result}"
