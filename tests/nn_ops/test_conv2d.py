"""Test conv2d equivalence."""

import pytest

from cnake_charmer.cy.nn_ops.conv2d import conv2d as cy_func
from cnake_charmer.py.nn_ops.conv2d import conv2d as py_func


@pytest.mark.parametrize("n", [5, 10, 50, 100])
def test_conv2d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
