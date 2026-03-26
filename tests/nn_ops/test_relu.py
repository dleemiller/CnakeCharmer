"""Test relu equivalence."""

import pytest

from cnake_charmer.cy.nn_ops.relu import relu as cy_func
from cnake_charmer.py.nn_ops.relu import relu as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_relu_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
