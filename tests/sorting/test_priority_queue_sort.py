"""Test priority_queue_sort equivalence."""

import pytest

from cnake_charmer.cy.sorting.priority_queue_sort import priority_queue_sort as cy_func
from cnake_charmer.py.sorting.priority_queue_sort import priority_queue_sort as py_func


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_priority_queue_sort_equivalence(n):
    assert py_func(n) == cy_func(n)
