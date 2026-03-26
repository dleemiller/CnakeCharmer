"""Test cross_entropy equivalence."""

import pytest

from cnake_charmer.cy.nn_ops.cross_entropy import cross_entropy as cy_func
from cnake_charmer.py.nn_ops.cross_entropy import cross_entropy as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_cross_entropy_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # f32 Cython vs f64 Python — relative tolerance
    assert abs(py_result - cy_result) / max(abs(py_result), 1.0) < 1e-4
