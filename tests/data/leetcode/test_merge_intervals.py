"""Test merge_intervals equivalence."""

import pytest

from cnake_data.cy.leetcode.merge_intervals import merge_intervals as cy_func
from cnake_data.py.leetcode.merge_intervals import merge_intervals as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_merge_intervals_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
