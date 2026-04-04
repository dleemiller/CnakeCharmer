"""Test particle_step_energy_class equivalence."""

import pytest

from cnake_data.cy.simulation.particle_step_energy_class import (
    particle_step_energy_class as cy_func,
)
from cnake_data.py.simulation.particle_step_energy_class import (
    particle_step_energy_class as py_func,
)


@pytest.mark.parametrize(
    "n,steps,dt,force_scale",
    [(20, 120, 0.01, 1.1), (30, 200, 0.02, 0.9), (25, 160, 0.015, 1.3)],
)
def test_particle_step_energy_class_equivalence(n, steps, dt, force_scale):
    py_result = py_func(n, steps, dt, force_scale)
    cy_result = cy_func(n, steps, dt, force_scale)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-8
