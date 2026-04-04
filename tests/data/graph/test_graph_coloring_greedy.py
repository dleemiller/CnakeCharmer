"""Test graph_coloring_greedy equivalence."""

import pytest

from cnake_data.cy.graph.graph_coloring_greedy import graph_coloring_greedy as cy_func
from cnake_data.py.graph.graph_coloring_greedy import graph_coloring_greedy as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_graph_coloring_greedy_equivalence(n):
    assert py_func(n) == cy_func(n)
