"""Test camera_projection equivalence."""

import pytest

from cnake_data.cy.numerical.camera_projection import camera_projection as cy_func
from cnake_data.py.numerical.camera_projection import camera_projection as py_func


@pytest.mark.parametrize(
    "n_pts",
    [200, 500, 1000, 2000],
)
def test_camera_projection_equivalence(n_pts):
    assert py_func(n_pts) == cy_func(n_pts)
