"""Test moving window sum equivalence."""

import pytest

from cnake_charmer.cy.numerical.moving_window_sum import moving_window_sum as cy_moving_window_sum
from cnake_charmer.py.numerical.moving_window_sum import moving_window_sum as py_moving_window_sum


@pytest.mark.parametrize("n", [200, 1000, 10000, 100000])
def test_moving_window_sum_equivalence(n):
    assert py_moving_window_sum(n) == cy_moving_window_sum(n)
