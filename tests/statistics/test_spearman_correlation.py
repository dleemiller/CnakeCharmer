"""Test spearman_correlation equivalence."""

import pytest

from cnake_charmer.cy.statistics.spearman_correlation import spearman_correlation as cy_func
from cnake_charmer.py.statistics.spearman_correlation import spearman_correlation as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_spearman_correlation_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
