"""Test heap_sort equivalence."""

import pytest

from cnake_charmer.cy.algorithms.heap_sort import heap_sort as cy_heap_sort
from cnake_charmer.py.algorithms.heap_sort import heap_sort as py_heap_sort


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_heap_sort_equivalence(n):
    py_result = py_heap_sort(n)
    cy_result = cy_heap_sort(n)
    assert py_result == cy_result, f"Mismatch at n={n}"
