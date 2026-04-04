"""Test topological_sort_kahn equivalence."""

import pytest

from cnake_data.cy.graph.topological_sort_kahn import topological_sort_kahn as cy_func
from cnake_data.py.graph.topological_sort_kahn import topological_sort_kahn as py_func


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_topological_sort_kahn_equivalence(n):
    assert py_func(n) == cy_func(n)
