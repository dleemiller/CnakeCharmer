"""Test epidemic_sir equivalence."""

import pytest

from cnake_charmer.cy.simulation.epidemic_sir import epidemic_sir as cy_func
from cnake_charmer.py.simulation.epidemic_sir import epidemic_sir as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_epidemic_sir_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
