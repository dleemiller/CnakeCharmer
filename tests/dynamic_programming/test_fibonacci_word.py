"""Test fibonacci_word equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.fibonacci_word import fibonacci_word as cy_func
from cnake_charmer.py.dynamic_programming.fibonacci_word import fibonacci_word as py_func


@pytest.mark.parametrize("n", [1, 2, 5, 10, 100, 1000])
def test_fibonacci_word_equivalence(n):
    assert py_func(n) == cy_func(n)
