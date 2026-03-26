"""Test union_find equivalence."""

import pytest

from cnake_charmer.cy.algorithms.union_find import union_find as cy_union_find
from cnake_charmer.py.algorithms.union_find import union_find as py_union_find


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_union_find_equivalence(n):
    assert py_union_find(n) == cy_union_find(n)
