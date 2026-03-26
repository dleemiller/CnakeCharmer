"""Test binary search count equivalence."""

import pytest

from cnake_charmer.cy.algorithms.binary_search_count import (
    binary_search_count as cy_binary_search_count,
)
from cnake_charmer.py.algorithms.binary_search_count import (
    binary_search_count as py_binary_search_count,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_binary_search_count_equivalence(n):
    assert py_binary_search_count(n) == cy_binary_search_count(n)
