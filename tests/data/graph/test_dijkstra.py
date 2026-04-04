"""Test dijkstra equivalence."""

import pytest

from cnake_data.cy.graph.dijkstra import dijkstra as cy_func
from cnake_data.py.graph.dijkstra import dijkstra as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_dijkstra_equivalence(n):
    assert py_func(n) == cy_func(n)
