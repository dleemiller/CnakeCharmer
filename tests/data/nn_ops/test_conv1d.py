"""Test conv1d equivalence."""

import pytest

from cnake_data.cy.nn_ops.conv1d import conv1d as cy_func
from cnake_data.py.nn_ops.conv1d import conv1d as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_conv1d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
