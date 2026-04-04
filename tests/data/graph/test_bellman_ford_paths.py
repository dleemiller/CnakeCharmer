"""Test bellman_ford_paths equivalence."""

import pytest

from cnake_data.cy.graph.bellman_ford_paths import bellman_ford_paths as cy_func
from cnake_data.py.graph.bellman_ford_paths import bellman_ford_paths as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_bellman_ford_paths_equivalence(n):
    assert py_func(n) == cy_func(n)
