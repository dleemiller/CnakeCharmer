"""Test valid_parentheses_gen equivalence."""

import pytest

from cnake_data.cy.leetcode.valid_parentheses_gen import valid_parentheses_gen as cy_func
from cnake_data.py.leetcode.valid_parentheses_gen import valid_parentheses_gen as py_func


@pytest.mark.parametrize("n", [5, 10, 50, 100])
def test_valid_parentheses_gen_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
