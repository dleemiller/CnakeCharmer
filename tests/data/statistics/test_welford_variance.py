"""Test welford_variance equivalence."""

import pytest

from cnake_data.cy.statistics.welford_variance import welford_variance as cy_func
from cnake_data.py.statistics.welford_variance import welford_variance as py_func


@pytest.mark.parametrize("n", [1000, 10000, 100000, 500000])
def test_welford_variance_equivalence(n):
    py_mean, py_var = py_func(n)
    cy_mean, cy_var = cy_func(n)
    assert abs(py_mean - cy_mean) / max(abs(py_mean), 1e-10) < 1e-10
    assert abs(py_var - cy_var) / max(abs(py_var), 1e-10) < 1e-10
