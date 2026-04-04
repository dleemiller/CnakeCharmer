"""Test bootstrap_mean equivalence."""

import pytest

from cnake_data.cy.statistics.bootstrap_mean import bootstrap_mean as cy_func
from cnake_data.py.statistics.bootstrap_mean import bootstrap_mean as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_bootstrap_mean_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
