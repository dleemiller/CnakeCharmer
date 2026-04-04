"""Test kruskal_mst equivalence."""

import pytest

from cnake_data.cy.graph.kruskal_mst import kruskal_mst as cy_func
from cnake_data.py.graph.kruskal_mst import kruskal_mst as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_kruskal_mst_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
