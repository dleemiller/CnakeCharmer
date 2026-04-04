"""Test bipartite_check equivalence."""

import pytest

from cnake_data.cy.graph.bipartite_check import bipartite_check as cy_func
from cnake_data.py.graph.bipartite_check import bipartite_check as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_bipartite_check_equivalence(n):
    assert py_func(n) == cy_func(n)
