"""Test reaction_diffusion equivalence."""

import pytest

from cnake_data.cy.simulation.reaction_diffusion import reaction_diffusion as cy_func
from cnake_data.py.simulation.reaction_diffusion import reaction_diffusion as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_reaction_diffusion_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
