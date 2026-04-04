"""Test floyd_warshall_apsp equivalence."""

import pytest

from cnake_data.cy.graph.floyd_warshall_apsp import floyd_warshall_apsp as cy_func
from cnake_data.py.graph.floyd_warshall_apsp import floyd_warshall_apsp as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_floyd_warshall_apsp_equivalence(n):
    assert py_func(n) == cy_func(n)
