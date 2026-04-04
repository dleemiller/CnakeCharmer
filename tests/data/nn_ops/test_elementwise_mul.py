"""Test elementwise_mul equivalence."""

import pytest

from cnake_data.cy.nn_ops.elementwise_mul import elementwise_mul as cy_func
from cnake_data.py.nn_ops.elementwise_mul import elementwise_mul as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_elementwise_mul_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < abs(py_result) * 1e-6 + 1e-6
