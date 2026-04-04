"""Test rowwise_kronecker equivalence."""

import pytest

from cnake_data.cy.numerical.rowwise_kronecker import rowwise_kronecker as cy_func
from cnake_data.py.numerical.rowwise_kronecker import rowwise_kronecker as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_rowwise_kronecker_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
