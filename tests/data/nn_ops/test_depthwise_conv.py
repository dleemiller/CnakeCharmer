"""Test depthwise_conv equivalence."""

import pytest

from cnake_data.cy.nn_ops.depthwise_conv import depthwise_conv as cy_func
from cnake_data.py.nn_ops.depthwise_conv import depthwise_conv as py_func


@pytest.mark.parametrize("n", [16, 160, 1600, 16000])
def test_depthwise_conv_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # f32 Cython vs f64 Python — relative tolerance
    assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-4
