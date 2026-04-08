"""Test lambertian_shading equivalence."""

import pytest

from cnake_data.cy.physics.lambertian_shading import lambertian_shading as cy_func
from cnake_data.py.physics.lambertian_shading import lambertian_shading as py_func


@pytest.mark.parametrize(
    "rows,cols",
    [
        (75, 75),
        (150, 150),
        (100, 120),
        (80, 80),
    ],
)
def test_lambertian_shading_equivalence(rows, cols):
    assert py_func(rows, cols) == cy_func(rows, cols)
