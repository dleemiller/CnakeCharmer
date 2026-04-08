"""Test phong_shading equivalence."""

import pytest

from cnake_data.cy.physics.phong_shading import phong_shading as cy_func
from cnake_data.py.physics.phong_shading import phong_shading as py_func


@pytest.mark.parametrize(
    "rows,cols",
    [
        (75, 75),
        (150, 150),
        (100, 120),
        (80, 80),
    ],
)
def test_phong_shading_equivalence(rows, cols):
    assert py_func(rows, cols) == cy_func(rows, cols)
