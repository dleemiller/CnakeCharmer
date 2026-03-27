"""Test pythran_fused_expr equivalence."""

import pytest

from cnake_charmer.cy.numerical.pythran_fused_expr import pythran_fused_expr as cy_func
from cnake_charmer.py.numerical.pythran_fused_expr import pythran_fused_expr as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_pythran_fused_expr_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-6, abs(py_result) * 1e-9)
    assert abs(py_result - cy_result) < tol
