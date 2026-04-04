"""Test mean_squared_error equivalence."""

import pytest

from cnake_data.cy.statistics.mean_squared_error import mean_squared_error as cy_func
from cnake_data.py.statistics.mean_squared_error import mean_squared_error as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_mean_squared_error_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
