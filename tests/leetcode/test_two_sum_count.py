"""Test two_sum_count equivalence."""

import pytest

from cnake_charmer.cy.leetcode.two_sum_count import two_sum_count as cy_func
from cnake_charmer.py.leetcode.two_sum_count import two_sum_count as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_two_sum_count_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
