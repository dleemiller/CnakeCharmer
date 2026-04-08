"""Test bicubic_interp equivalence."""

import pytest

from cnake_data.cy.image_processing.bicubic_interp import bicubic_interp as cy_func
from cnake_data.py.image_processing.bicubic_interp import bicubic_interp as py_func


@pytest.mark.parametrize(
    "src_rows,src_cols,scale",
    [
        (40, 40, 2),
        (80, 80, 2),
        (30, 30, 3),
        (20, 20, 2),
    ],
)
def test_bicubic_interp_equivalence(src_rows, src_cols, scale):
    assert py_func(src_rows, src_cols, scale) == cy_func(src_rows, src_cols, scale)
