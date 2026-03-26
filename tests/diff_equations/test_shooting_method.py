"""Test shooting_method equivalence."""

import pytest

from cnake_charmer.cy.diff_equations.shooting_method import shooting_method as cy_func
from cnake_charmer.py.diff_equations.shooting_method import shooting_method as py_func


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_shooting_method_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
