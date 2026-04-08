"""Test radial_distortion equivalence."""

import pytest

from cnake_data.cy.image_processing.radial_distortion import radial_distortion as cy_func
from cnake_data.py.image_processing.radial_distortion import radial_distortion as py_func


@pytest.mark.parametrize(
    "rows,cols",
    [
        (100, 100),
        (200, 200),
        (150, 200),
        (80, 80),
    ],
)
def test_radial_distortion_equivalence(rows, cols):
    assert py_func(rows, cols) == cy_func(rows, cols)
