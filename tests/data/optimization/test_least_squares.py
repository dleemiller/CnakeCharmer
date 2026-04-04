"""Test least_squares equivalence."""

import math

import pytest

from cnake_data.cy.optimization.least_squares import least_squares as cy_func
from cnake_data.py.optimization.least_squares import least_squares as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_least_squares_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    if math.isnan(py_result) and math.isnan(cy_result):
        return  # both diverged identically
    assert abs(py_result - cy_result) < 1e-6
