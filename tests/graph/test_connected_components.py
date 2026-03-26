"""Test connected_components equivalence."""

import pytest

from cnake_charmer.cy.graph.connected_components import connected_components as cy_func
from cnake_charmer.py.graph.connected_components import connected_components as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_connected_components_equivalence(n):
    assert py_func(n) == cy_func(n)
