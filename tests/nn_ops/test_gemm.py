"""Test gemm equivalence."""

import pytest

from cnake_charmer.cy.nn_ops.gemm import gemm as cy_func
from cnake_charmer.py.nn_ops.gemm import gemm as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_gemm_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # f32 accumulation vs f64 — relative tolerance
    assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-3
