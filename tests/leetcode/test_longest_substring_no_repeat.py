"""Test longest_substring_no_repeat equivalence."""

import pytest

from cnake_charmer.cy.leetcode.longest_substring_no_repeat import (
    longest_substring_no_repeat as cy_func,
)
from cnake_charmer.py.leetcode.longest_substring_no_repeat import (
    longest_substring_no_repeat as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_longest_substring_no_repeat_equivalence(n):
    assert py_func(n) == cy_func(n)
