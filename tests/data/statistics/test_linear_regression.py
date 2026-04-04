"""Test linear_regression equivalence."""

import pytest

from cnake_data.cy.statistics.linear_regression import linear_regression as cy_func
from cnake_data.py.statistics.linear_regression import linear_regression as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_linear_regression_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
