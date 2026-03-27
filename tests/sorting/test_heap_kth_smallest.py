"""Test heap_kth_smallest equivalence."""

import pytest

from cnake_charmer.cy.sorting.heap_kth_smallest import heap_kth_smallest as cy_func
from cnake_charmer.py.sorting.heap_kth_smallest import heap_kth_smallest as py_func


@pytest.mark.parametrize("n", [200, 1000, 5000, 10000])
def test_heap_kth_smallest_equivalence(n):
    assert py_func(n) == cy_func(n)
