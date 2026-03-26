"""Test weighted_percentile equivalence."""

import pytest

from cnake_charmer.cy.statistics.weighted_percentile import weighted_percentile as cy_func
from cnake_charmer.py.statistics.weighted_percentile import weighted_percentile as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_weighted_percentile_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
