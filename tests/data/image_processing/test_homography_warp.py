"""Test homography_warp equivalence."""

import pytest

from cnake_data.cy.image_processing.homography_warp import homography_warp as cy_func
from cnake_data.py.image_processing.homography_warp import homography_warp as py_func


@pytest.mark.parametrize(
    "rows,cols",
    [
        (60, 60),
        (120, 120),
        (80, 100),
        (150, 100),
    ],
)
def test_homography_warp_equivalence(rows, cols):
    assert py_func(rows, cols) == cy_func(rows, cols)
