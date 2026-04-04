"""Test global_avg_pool equivalence."""

import pytest

from cnake_data.cy.nn_ops.global_avg_pool import global_avg_pool as cy_func
from cnake_data.py.nn_ops.global_avg_pool import global_avg_pool as py_func


@pytest.mark.parametrize("n", [64, 640, 6400, 64000])
def test_global_avg_pool_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # f32 Cython vs f64 Python — relative tolerance
    assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-4
