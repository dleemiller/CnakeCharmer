"""Test 2D prefix sum equivalence."""

import pytest

from cnake_charmer.cy.numerical.prefix_sum_2d import prefix_sum_2d as cy_prefix_sum_2d
from cnake_charmer.py.numerical.prefix_sum_2d import prefix_sum_2d as py_prefix_sum_2d


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_prefix_sum_2d_equivalence(n):
    assert py_prefix_sum_2d(n) == cy_prefix_sum_2d(n)
