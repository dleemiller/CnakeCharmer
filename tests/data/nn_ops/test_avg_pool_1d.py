"""Test avg_pool_1d equivalence."""

import pytest

from cnake_data.cy.nn_ops.avg_pool_1d import avg_pool_1d as cy_func
from cnake_data.py.nn_ops.avg_pool_1d import avg_pool_1d as py_func


@pytest.mark.parametrize("n", [16, 100, 1000, 10000])
def test_avg_pool_1d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # f32 Cython vs f64 Python — relative tolerance
    assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-4
