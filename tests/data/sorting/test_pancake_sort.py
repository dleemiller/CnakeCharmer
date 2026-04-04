"""Test pancake_sort equivalence."""

import pytest

from cnake_data.cy.sorting.pancake_sort import pancake_sort as cy_func
from cnake_data.py.sorting.pancake_sort import pancake_sort as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_pancake_sort_equivalence(n):
    assert py_func(n) == cy_func(n)
