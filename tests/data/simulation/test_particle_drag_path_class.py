"""Test particle_drag_path_class equivalence."""

import pytest

from cnake_data.cy.simulation.particle_drag_path_class import (
    particle_drag_path_class as cy_func,
)
from cnake_data.py.simulation.particle_drag_path_class import (
    particle_drag_path_class as py_func,
)


@pytest.mark.parametrize(
    "n,steps,dt,drag", [(20, 80, 0.01, 0.02), (24, 100, 0.008, 0.03), (18, 90, 0.012, 0.04)]
)
def test_particle_drag_path_class_equivalence(n, steps, dt, drag):
    py_result = py_func(n, steps, dt, drag)
    cy_result = cy_func(n, steps, dt, drag)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-8
