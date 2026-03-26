"""Test max_subarray equivalence."""

import pytest

from cnake_charmer.cy.leetcode.max_subarray import max_subarray as cy_func
from cnake_charmer.py.leetcode.max_subarray import max_subarray as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_max_subarray_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
