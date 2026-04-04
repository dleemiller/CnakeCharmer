"""Test valid_parentheses equivalence."""

import pytest

from cnake_data.cy.leetcode.valid_parentheses import valid_parentheses as cy_func
from cnake_data.py.leetcode.valid_parentheses import valid_parentheses as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_valid_parentheses_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
