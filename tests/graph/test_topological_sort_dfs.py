"""Test topological_sort_dfs equivalence."""

import pytest

from cnake_charmer.cy.graph.topological_sort_dfs import topological_sort_dfs as cy_func
from cnake_charmer.py.graph.topological_sort_dfs import topological_sort_dfs as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_topological_sort_dfs_equivalence(n):
    assert py_func(n) == cy_func(n)
