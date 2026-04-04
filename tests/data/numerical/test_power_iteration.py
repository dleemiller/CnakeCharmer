"""Test power_iteration equivalence."""

import pytest

from cnake_data.cy.numerical.power_iteration import power_iteration as cy_func
from cnake_data.py.numerical.power_iteration import power_iteration as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_power_iteration_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
