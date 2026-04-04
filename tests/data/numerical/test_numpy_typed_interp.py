"""Test numpy_typed_interp equivalence."""

import pytest

from cnake_data.cy.numerical.numpy_typed_interp import (
    numpy_typed_interp as cy_func,
)
from cnake_data.py.numerical.numpy_typed_interp import (
    numpy_typed_interp as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_numpy_typed_interp_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-3, abs(py_result) * 1e-6)
    assert abs(py_result - cy_result) < tol, f"Mismatch: py={py_result}, cy={cy_result}"
