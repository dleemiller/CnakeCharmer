"""Test thin_plate_spline equivalence."""

import pytest

from cnake_data.cy.image_processing.thin_plate_spline import thin_plate_spline as cy_func
from cnake_data.py.image_processing.thin_plate_spline import thin_plate_spline as py_func


@pytest.mark.parametrize(
    "rows,cols,n_ctrl",
    [
        (30, 30, 9),
        (60, 60, 12),
        (40, 40, 16),
        (50, 50, 9),
    ],
)
def test_thin_plate_spline_equivalence(rows, cols, n_ctrl):
    assert py_func(rows, cols, n_ctrl) == cy_func(rows, cols, n_ctrl)
