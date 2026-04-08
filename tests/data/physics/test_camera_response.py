"""Test camera_response equivalence."""

import pytest

from cnake_data.cy.physics.camera_response import camera_response as cy_func
from cnake_data.py.physics.camera_response import camera_response as py_func


@pytest.mark.parametrize(
    "rows,cols",
    [
        (75, 75),
        (150, 150),
        (100, 120),
        (80, 80),
    ],
)
def test_camera_response_equivalence(rows, cols):
    assert py_func(rows, cols) == cy_func(rows, cols)
