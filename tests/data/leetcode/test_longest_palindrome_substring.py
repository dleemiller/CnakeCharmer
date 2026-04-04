"""Test longest_palindrome_substring equivalence."""

import pytest

from cnake_data.cy.leetcode.longest_palindrome_substring import (
    longest_palindrome_substring as cy_func,
)
from cnake_data.py.leetcode.longest_palindrome_substring import (
    longest_palindrome_substring as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_longest_palindrome_substring_equivalence(n):
    assert py_func(n) == cy_func(n)
