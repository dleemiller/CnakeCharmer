"""Test collision_grid_markers_class equivalence."""

import pytest

from cnake_charmer.cy.simulation.collision_grid_markers_class import (
    collision_grid_markers_class as cy_func,
)
from cnake_charmer.py.simulation.collision_grid_markers_class import (
    collision_grid_markers_class as py_func,
)


@pytest.mark.parametrize(
    "width,height,block,n_particles,steps,seed",
    [(48, 48, 4, 80, 20, 3), (60, 72, 6, 120, 25, 9), (96, 84, 6, 160, 18, 17)],
)
def test_collision_grid_markers_class_equivalence(width, height, block, n_particles, steps, seed):
    assert py_func(width, height, block, n_particles, steps, seed) == cy_func(
        width, height, block, n_particles, steps, seed
    )
