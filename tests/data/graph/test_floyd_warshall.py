"""Test floyd_warshall equivalence."""

import pytest

from cnake_data.cy.graph.floyd_warshall import floyd_warshall as cy_func
from cnake_data.py.graph.floyd_warshall import floyd_warshall as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_floyd_warshall_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
