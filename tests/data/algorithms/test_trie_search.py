"""Test trie search equivalence."""

import pytest

from cnake_data.cy.algorithms.trie_search import trie_search as cy_trie_search
from cnake_data.py.algorithms.trie_search import trie_search as py_trie_search


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_trie_search_equivalence(n):
    assert py_trie_search(n) == cy_trie_search(n)
