"""Test particle_bounce equivalence."""

import pytest

from cnake_charmer.cy.simulation.particle_bounce import particle_bounce as cy_func
from cnake_charmer.py.simulation.particle_bounce import particle_bounce as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_particle_bounce_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-3, abs(py_result) * 1e-9), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
