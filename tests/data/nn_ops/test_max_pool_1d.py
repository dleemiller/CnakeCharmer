"""Test max_pool_1d equivalence."""

import pytest

from cnake_data.cy.nn_ops.max_pool_1d import max_pool_1d as cy_func
from cnake_data.py.nn_ops.max_pool_1d import max_pool_1d as py_func


@pytest.mark.parametrize("n", [8, 100, 1000, 10000])
def test_max_pool_1d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
