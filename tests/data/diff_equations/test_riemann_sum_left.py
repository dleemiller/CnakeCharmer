"""Test riemann_sum_left equivalence."""

import pytest

from cnake_data.cy.diff_equations.riemann_sum_left import riemann_sum_left as cy_func
from cnake_data.py.diff_equations.riemann_sum_left import riemann_sum_left as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_riemann_sum_left_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-9, f"Mismatch: py={py_result}, cy={cy_result}"
