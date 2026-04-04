"""Test numpy_prange_distance equivalence."""

import pytest

from cnake_data.cy.geometry.numpy_prange_distance import (
    numpy_prange_distance as cy_func,
)
from cnake_data.py.geometry.numpy_prange_distance import (
    numpy_prange_distance as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500])
def test_numpy_prange_distance_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-3, abs(py_result) * 1e-6)
    assert abs(py_result - cy_result) < tol, f"Mismatch: py={py_result}, cy={cy_result}"
