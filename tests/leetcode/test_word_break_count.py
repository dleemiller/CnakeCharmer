"""Test word_break_count equivalence."""

import pytest

from cnake_charmer.cy.leetcode.word_break_count import word_break_count as cy_func
from cnake_charmer.py.leetcode.word_break_count import word_break_count as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_word_break_count_equivalence(n):
    assert py_func(n) == cy_func(n)
