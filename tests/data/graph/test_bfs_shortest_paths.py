"""Test bfs_shortest_paths equivalence."""

import pytest

from cnake_data.cy.graph.bfs_shortest_paths import bfs_shortest_paths as cy_bfs_shortest_paths
from cnake_data.py.graph.bfs_shortest_paths import bfs_shortest_paths as py_bfs_shortest_paths


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_bfs_shortest_paths_equivalence(n):
    py_result = py_bfs_shortest_paths(n)
    cy_result = cy_bfs_shortest_paths(n)
    assert py_result == cy_result, f"Mismatch at n={n}: py={py_result}, cy={cy_result}"
