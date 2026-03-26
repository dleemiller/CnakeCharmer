"""Test tim_sort_merge equivalence."""

import pytest

from cnake_charmer.cy.sorting.tim_sort_merge import tim_sort_merge as cy_func
from cnake_charmer.py.sorting.tim_sort_merge import tim_sort_merge as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_tim_sort_merge_equivalence(n):
    assert py_func(n) == cy_func(n)
