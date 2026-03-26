"""Test instance_norm equivalence."""

import pytest

from cnake_charmer.cy.nn_ops.instance_norm import instance_norm as cy_func
from cnake_charmer.py.nn_ops.instance_norm import instance_norm as py_func


@pytest.mark.parametrize("n", [16, 160, 1600, 16000])
def test_instance_norm_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # f32 Cython vs f64 Python — absolute tolerance (result near zero)
    assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-4
