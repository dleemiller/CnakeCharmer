"""Test running_variance equivalence."""

import pytest

from cnake_charmer.cy.statistics.running_variance import running_variance as cy_func
from cnake_charmer.py.statistics.running_variance import running_variance as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_running_variance_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
