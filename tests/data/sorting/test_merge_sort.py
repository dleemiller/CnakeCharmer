"""Test merge sort equivalence."""

import pytest

from cnake_data.cy.sorting.merge_sort import merge_sort as cy_merge_sort
from cnake_data.py.sorting.merge_sort import merge_sort as py_merge_sort


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_merge_sort_equivalence(n):
    assert py_merge_sort(n) == cy_merge_sort(n)
