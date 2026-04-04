"""Test numpy_cross_entropy equivalence."""

import pytest

from cnake_data.cy.nn_ops.numpy_cross_entropy import (
    numpy_cross_entropy as cy_func,
)
from cnake_data.py.nn_ops.numpy_cross_entropy import (
    numpy_cross_entropy as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_numpy_cross_entropy_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-3, abs(py_result) * 1e-6)
    assert abs(py_result - cy_result) < tol, f"Mismatch: py={py_result}, cy={cy_result}"
