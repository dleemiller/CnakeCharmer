"""Test counting_sort_chars equivalence."""

import pytest

from cnake_charmer.cy.sorting.counting_sort_chars import counting_sort_chars as cy_func
from cnake_charmer.py.sorting.counting_sort_chars import counting_sort_chars as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_counting_sort_chars_equivalence(n):
    assert py_func(n) == cy_func(n)
