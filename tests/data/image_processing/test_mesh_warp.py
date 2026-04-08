"""Test mesh_warp equivalence."""

import pytest

from cnake_data.cy.image_processing.mesh_warp import mesh_warp as cy_func
from cnake_data.py.image_processing.mesh_warp import mesh_warp as py_func


@pytest.mark.parametrize(
    "rows,cols,mesh_size",
    [
        (50, 50, 6),
        (100, 100, 8),
        (80, 80, 5),
        (60, 60, 6),
    ],
)
def test_mesh_warp_equivalence(rows, cols, mesh_size):
    assert py_func(rows, cols, mesh_size) == cy_func(rows, cols, mesh_size)
