"""Test coin change equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.coin_change import coin_change as cy_coin_change
from cnake_data.py.dynamic_programming.coin_change import coin_change as py_coin_change


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_coin_change_equivalence(n):
    assert py_coin_change(n) == cy_coin_change(n)
