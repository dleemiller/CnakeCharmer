"""Test layer_norm equivalence."""

import pytest

from cnake_data.cy.nn_ops.layer_norm import layer_norm as cy_func
from cnake_data.py.nn_ops.layer_norm import layer_norm as py_func


@pytest.mark.parametrize("n", [64, 640, 6400, 64000])
def test_layer_norm_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
