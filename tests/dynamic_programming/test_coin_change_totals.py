"""Test coin_change_totals equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.coin_change_totals import coin_change_totals as cy_func
from cnake_charmer.py.dynamic_programming.coin_change_totals import coin_change_totals as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_coin_change_totals_equivalence(n):
    assert py_func(n) == cy_func(n)
