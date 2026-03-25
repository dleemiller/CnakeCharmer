"""Test selection sort equivalence."""

import pytest

from cnake_charmer.cy.algorithms.selection_sort import selection_sort as cy_selection_sort
from cnake_charmer.py.algorithms.selection_sort import selection_sort as py_selection_sort


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_selection_sort_equivalence(n):
    assert py_selection_sort(n) == cy_selection_sort(n)
