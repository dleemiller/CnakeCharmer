"""Test lagrange_interpolation equivalence."""

import pytest

from cnake_data.cy.numerical.lagrange_interpolation import lagrange_interpolation as cy_func
from cnake_data.py.numerical.lagrange_interpolation import lagrange_interpolation as py_func


@pytest.mark.parametrize("n", [2, 10, 50, 100])
def test_lagrange_interpolation_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch at n={n}: {py_result} vs {cy_result}"
