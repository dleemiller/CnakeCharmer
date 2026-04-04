"""Test gradient_descent equivalence."""

import pytest

from cnake_data.cy.optimization.gradient_descent import gradient_descent as cy_func
from cnake_data.py.optimization.gradient_descent import gradient_descent as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_gradient_descent_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
