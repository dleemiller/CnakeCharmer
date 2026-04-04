"""Test cellular_automaton_1d equivalence."""

import pytest

from cnake_data.cy.simulation.cellular_automaton_1d import cellular_automaton_1d as cy_func
from cnake_data.py.simulation.cellular_automaton_1d import cellular_automaton_1d as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_cellular_automaton_1d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
