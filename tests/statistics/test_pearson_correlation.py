"""Test pearson_correlation equivalence."""

import pytest

from cnake_charmer.cy.statistics.pearson_correlation import pearson_correlation as cy_func
from cnake_charmer.py.statistics.pearson_correlation import pearson_correlation as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_pearson_correlation_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
