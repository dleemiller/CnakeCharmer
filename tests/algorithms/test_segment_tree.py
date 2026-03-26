"""Test segment_tree equivalence."""

import pytest

from cnake_charmer.cy.algorithms.segment_tree import segment_tree as cy_segment_tree
from cnake_charmer.py.algorithms.segment_tree import segment_tree as py_segment_tree


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_segment_tree_equivalence(n):
    assert py_segment_tree(n) == cy_segment_tree(n)
