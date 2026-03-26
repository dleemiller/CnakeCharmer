"""Test exponential_histogram equivalence."""

import pytest

from cnake_charmer.cy.statistics.exponential_histogram import exponential_histogram as cy_func
from cnake_charmer.py.statistics.exponential_histogram import exponential_histogram as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_exponential_histogram_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
