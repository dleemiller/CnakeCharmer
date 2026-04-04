"""Test numpy_l2_normalize equivalence."""

import pytest

from cnake_data.cy.nn_ops.numpy_l2_normalize import (
    numpy_l2_normalize as cy_func,
)
from cnake_data.py.nn_ops.numpy_l2_normalize import (
    numpy_l2_normalize as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500])
def test_numpy_l2_normalize_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-3, abs(py_result) * 1e-6)
    assert abs(py_result - cy_result) < tol, f"Mismatch: py={py_result}, cy={cy_result}"
