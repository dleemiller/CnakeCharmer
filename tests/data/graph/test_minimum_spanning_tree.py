"""Test minimum_spanning_tree equivalence."""

import pytest

from cnake_data.cy.graph.minimum_spanning_tree import minimum_spanning_tree as cy_func
from cnake_data.py.graph.minimum_spanning_tree import minimum_spanning_tree as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_minimum_spanning_tree_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
