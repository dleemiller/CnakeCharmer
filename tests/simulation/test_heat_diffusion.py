"""Test heat_diffusion equivalence."""

import pytest

from cnake_charmer.cy.simulation.heat_diffusion import heat_diffusion as cy_func
from cnake_charmer.py.simulation.heat_diffusion import heat_diffusion as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_heat_diffusion_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
