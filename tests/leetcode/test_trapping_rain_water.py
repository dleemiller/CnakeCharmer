"""Test trapping_rain_water equivalence."""

import pytest

from cnake_charmer.cy.leetcode.trapping_rain_water import trapping_rain_water as cy_func
from cnake_charmer.py.leetcode.trapping_rain_water import trapping_rain_water as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_trapping_rain_water_equivalence(n):
    assert py_func(n) == cy_func(n)
