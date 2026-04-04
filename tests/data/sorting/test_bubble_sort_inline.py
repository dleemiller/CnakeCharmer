"""Test bubble_sort_inline equivalence."""

import pytest

from cnake_data.cy.sorting.bubble_sort_inline import bubble_sort_sum as cy_bubble_sort_sum
from cnake_data.py.sorting.bubble_sort_inline import bubble_sort_sum as py_bubble_sort_sum


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_bubble_sort_sum_equivalence(n):
    assert py_bubble_sort_sum(n) == cy_bubble_sort_sum(n)
