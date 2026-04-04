"""Test silu equivalence."""

import pytest

from cnake_data.cy.nn_ops.silu import silu as cy_func
from cnake_data.py.nn_ops.silu import silu as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_silu_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # f32 Cython vs f64 Python — relative tolerance
    assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-4
