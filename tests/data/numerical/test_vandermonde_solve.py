"""Test vandermonde_solve equivalence."""

import pytest

from cnake_data.cy.numerical.vandermonde_solve import vandermonde_solve as cy_func
from cnake_data.py.numerical.vandermonde_solve import vandermonde_solve as py_func


@pytest.mark.parametrize("n", [10, 30, 50, 80])
def test_vandermonde_solve_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={p}, cy={c}"
