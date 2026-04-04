"""Test numpy_bincount_weighted equivalence."""

import pytest

from cnake_data.cy.statistics.numpy_bincount_weighted import (
    numpy_bincount_weighted as cy_func,
)
from cnake_data.py.statistics.numpy_bincount_weighted import (
    numpy_bincount_weighted as py_func,
)


@pytest.mark.parametrize("n", [1000, 10000, 50000])
def test_numpy_bincount_weighted_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-3, abs(py_result) * 1e-6)
    assert abs(py_result - cy_result) < tol, f"Mismatch: py={py_result}, cy={cy_result}"
