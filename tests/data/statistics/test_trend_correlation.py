"""Test trend_correlation equivalence."""

import pytest

from cnake_data.cy.statistics.trend_correlation import trend_correlation as cy_func
from cnake_data.py.statistics.trend_correlation import trend_correlation as py_func


@pytest.mark.parametrize("n", [5, 20, 50, 100])
def test_trend_correlation_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result[0] - cy_result[0]) < 1e-6
    assert abs(py_result[1] - cy_result[1]) < 1e-6
