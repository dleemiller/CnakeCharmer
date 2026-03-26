"""Test brownian_motion equivalence."""

import pytest

from cnake_charmer.cy.simulation.brownian_motion import brownian_motion as cy_func
from cnake_charmer.py.simulation.brownian_motion import brownian_motion as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_brownian_motion_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
