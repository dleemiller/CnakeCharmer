"""Test shortest_path_dag equivalence."""

import pytest

from cnake_charmer.cy.graph.shortest_path_dag import shortest_path_dag as cy_func
from cnake_charmer.py.graph.shortest_path_dag import shortest_path_dag as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_shortest_path_dag_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
