"""Test two_sum_all_pairs equivalence."""

import pytest

from cnake_data.cy.leetcode.two_sum_all_pairs import two_sum_all_pairs as cy_func
from cnake_data.py.leetcode.two_sum_all_pairs import two_sum_all_pairs as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_two_sum_all_pairs_equivalence(n):
    assert py_func(n) == cy_func(n)
