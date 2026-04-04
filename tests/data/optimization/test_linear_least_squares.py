"""Test linear_least_squares equivalence."""

import pytest

from cnake_data.cy.optimization.linear_least_squares import linear_least_squares as cy_func
from cnake_data.py.optimization.linear_least_squares import linear_least_squares as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_linear_least_squares_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
