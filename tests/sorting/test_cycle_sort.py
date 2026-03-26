"""Test cycle_sort equivalence."""

import pytest

from cnake_charmer.cy.sorting.cycle_sort import cycle_sort as cy_func
from cnake_charmer.py.sorting.cycle_sort import cycle_sort as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_cycle_sort_equivalence(n):
    assert py_func(n) == cy_func(n)
