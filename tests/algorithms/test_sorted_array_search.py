"""Test sorted_array_search equivalence."""

import pytest

from cnake_charmer.cy.algorithms.sorted_array_search import sorted_array_search as cy_func
from cnake_charmer.py.algorithms.sorted_array_search import sorted_array_search as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_sorted_array_search_equivalence(n):
    assert py_func(n) == cy_func(n)
