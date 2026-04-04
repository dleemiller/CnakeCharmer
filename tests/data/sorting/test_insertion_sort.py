"""Test insertion sort equivalence."""

import pytest

from cnake_data.cy.sorting.insertion_sort import insertion_sort as cy_insertion_sort
from cnake_data.py.sorting.insertion_sort import insertion_sort as py_insertion_sort


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_insertion_sort_equivalence(n):
    assert py_insertion_sort(n) == cy_insertion_sort(n)
