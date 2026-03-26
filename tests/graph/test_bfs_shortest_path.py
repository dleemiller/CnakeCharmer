"""Test bfs_shortest_path equivalence."""

import pytest

from cnake_charmer.cy.graph.bfs_shortest_path import bfs_shortest_path as cy_func
from cnake_charmer.py.graph.bfs_shortest_path import bfs_shortest_path as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_bfs_shortest_path_equivalence(n):
    assert py_func(n) == cy_func(n)
