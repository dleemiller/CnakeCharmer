"""Test coin_change_count equivalence between Python and Cython."""

import pytest

from cnake_data.cy.dynamic_programming.coin_change_count import (
    coin_change_count as cy_coin_change_count,
)
from cnake_data.py.dynamic_programming.coin_change_count import (
    coin_change_count as py_coin_change_count,
)


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_coin_change_count_equivalence(n):
    assert py_coin_change_count(n) == cy_coin_change_count(n)
