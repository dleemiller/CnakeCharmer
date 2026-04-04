"""Test moving_median equivalence."""

import pytest

from cnake_data.cy.statistics.moving_median import moving_median as cy_func
from cnake_data.py.statistics.moving_median import moving_median as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_moving_median_equivalence(n):
    assert py_func(n) == cy_func(n)
