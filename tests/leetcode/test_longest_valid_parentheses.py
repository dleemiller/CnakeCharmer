"""Test longest_valid_parentheses equivalence."""

import pytest

from cnake_charmer.cy.leetcode.longest_valid_parentheses import longest_valid_parentheses as cy_func
from cnake_charmer.py.leetcode.longest_valid_parentheses import longest_valid_parentheses as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_longest_valid_parentheses_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
