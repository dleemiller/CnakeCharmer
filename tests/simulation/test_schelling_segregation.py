"""Test schelling_segregation equivalence."""

import pytest

from cnake_charmer.cy.simulation.schelling_segregation import schelling_segregation as cy_func
from cnake_charmer.py.simulation.schelling_segregation import schelling_segregation as py_func


@pytest.mark.parametrize("n", [30, 100, 300, 600])
def test_schelling_segregation_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
