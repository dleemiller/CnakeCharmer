"""Test levenberg_marquardt equivalence."""

import pytest

from cnake_data.cy.optimization.levenberg_marquardt import levenberg_marquardt as cy_func
from cnake_data.py.optimization.levenberg_marquardt import levenberg_marquardt as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_levenberg_marquardt_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
