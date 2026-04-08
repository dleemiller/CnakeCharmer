"""Test flat_field_correction equivalence."""

import pytest

from cnake_data.cy.image_processing.flat_field_correction import flat_field_correction as cy_func
from cnake_data.py.image_processing.flat_field_correction import flat_field_correction as py_func


@pytest.mark.parametrize(
    "rows,cols",
    [
        (100, 100),
        (200, 200),
        (150, 120),
        (80, 80),
    ],
)
def test_flat_field_correction_equivalence(rows, cols):
    assert py_func(rows, cols) == cy_func(rows, cols)
