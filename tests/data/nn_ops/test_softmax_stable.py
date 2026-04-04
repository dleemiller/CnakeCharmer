"""Test softmax_stable equivalence."""

import pytest

from cnake_data.cy.nn_ops.softmax_stable import softmax_stable as cy_func
from cnake_data.py.nn_ops.softmax_stable import softmax_stable as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_softmax_stable_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
