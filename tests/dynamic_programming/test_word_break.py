"""Test word_break equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.word_break import word_break as cy_func
from cnake_charmer.py.dynamic_programming.word_break import word_break as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_word_break_equivalence(n):
    assert py_func(n) == cy_func(n)
