"""Test matrix_power_trace equivalence."""

import pytest

from cnake_data.cy.numerical.matrix_power_trace import matrix_power_trace as cy_func
from cnake_data.py.numerical.matrix_power_trace import matrix_power_trace as py_func


@pytest.mark.parametrize("n", [5, 10, 30, 50])
def test_matrix_power_trace_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-3, abs(py_result) * 1e-6)
    assert abs(py_result - cy_result) < tol, f"Mismatch: py={py_result}, cy={cy_result}"
