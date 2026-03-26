"""Test fenwick_tree equivalence."""

import pytest

from cnake_charmer.cy.algorithms.fenwick_tree import fenwick_tree as cy_fenwick_tree
from cnake_charmer.py.algorithms.fenwick_tree import fenwick_tree as py_fenwick_tree


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_fenwick_tree_equivalence(n):
    assert py_fenwick_tree(n) == cy_fenwick_tree(n)
