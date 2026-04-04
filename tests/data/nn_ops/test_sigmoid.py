"""Test sigmoid equivalence."""

import pytest

from cnake_data.cy.nn_ops.sigmoid import sigmoid as cy_func
from cnake_data.py.nn_ops.sigmoid import sigmoid as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_sigmoid_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
