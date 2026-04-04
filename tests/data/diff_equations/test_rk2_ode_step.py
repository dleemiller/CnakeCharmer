"""Test rk2_ode_step equivalence."""

import pytest

from cnake_data.cy.diff_equations.rk2_ode_step import rk2_ode_step as cy_func
from cnake_data.py.diff_equations.rk2_ode_step import rk2_ode_step as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_rk2_ode_step_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
