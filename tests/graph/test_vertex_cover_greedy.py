"""Test vertex_cover_greedy equivalence."""

import pytest

from cnake_charmer.cy.graph.vertex_cover_greedy import vertex_cover_greedy as cy_func
from cnake_charmer.py.graph.vertex_cover_greedy import vertex_cover_greedy as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_vertex_cover_greedy_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
