"""Test bellman_ford equivalence."""

import pytest

from cnake_charmer.cy.graph.bellman_ford import bellman_ford as cy_func
from cnake_charmer.py.graph.bellman_ford import bellman_ford as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_bellman_ford_equivalence(n):
    assert py_func(n) == cy_func(n)
