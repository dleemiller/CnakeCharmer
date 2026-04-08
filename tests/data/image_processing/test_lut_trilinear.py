"""Test lut_trilinear equivalence."""

import pytest

from cnake_data.cy.image_processing.lut_trilinear import lut_trilinear as cy_func
from cnake_data.py.image_processing.lut_trilinear import lut_trilinear as py_func


@pytest.mark.parametrize(
    "rows,cols,lut_size",
    [
        (40, 40, 9),
        (80, 80, 17),
        (60, 60, 11),
        (30, 30, 9),
    ],
)
def test_lut_trilinear_equivalence(rows, cols, lut_size):
    assert py_func(rows, cols, lut_size) == cy_func(rows, cols, lut_size)
