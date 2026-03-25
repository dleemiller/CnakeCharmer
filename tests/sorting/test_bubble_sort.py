"""Test bubble sort equivalence."""

import pytest

from cnake_charmer.cy.sorting.bubble_sort import bubble_sort as cy_bubble_sort
from cnake_charmer.py.sorting.bubble_sort import bubble_sort as py_bubble_sort


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_bubble_sort_equivalence(n):
    assert py_bubble_sort(n) == cy_bubble_sort(n)
