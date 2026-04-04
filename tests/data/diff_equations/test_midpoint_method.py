"""Test midpoint_method equivalence."""

import pytest

from cnake_data.cy.diff_equations.midpoint_method import midpoint_method as cy_func
from cnake_data.py.diff_equations.midpoint_method import midpoint_method as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_midpoint_method_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
