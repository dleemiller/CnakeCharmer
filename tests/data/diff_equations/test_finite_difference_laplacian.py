"""Test finite_difference_laplacian equivalence."""

import pytest

from cnake_data.cy.diff_equations.finite_difference_laplacian import (
    finite_difference_laplacian as cy_func,
)
from cnake_data.py.diff_equations.finite_difference_laplacian import (
    finite_difference_laplacian as py_func,
)


@pytest.mark.parametrize("n", [10, 20, 50])
def test_finite_difference_laplacian_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
