"""Test ray_trace_sequential equivalence."""

import pytest

from cnake_data.cy.physics.ray_trace_sequential import ray_trace_sequential as cy_func
from cnake_data.py.physics.ray_trace_sequential import ray_trace_sequential as py_func


@pytest.mark.parametrize(
    "n_rays,n_surfaces",
    [
        (1000, 8),
        (5000, 8),
        (2000, 6),
        (3000, 10),
    ],
)
def test_ray_trace_sequential_equivalence(n_rays, n_surfaces):
    assert py_func(n_rays, n_surfaces) == cy_func(n_rays, n_surfaces)
