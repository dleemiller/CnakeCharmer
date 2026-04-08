"""Test uv_texture_sample equivalence."""

import pytest

from cnake_data.cy.image_processing.uv_texture_sample import uv_texture_sample as cy_func
from cnake_data.py.image_processing.uv_texture_sample import uv_texture_sample as py_func


@pytest.mark.parametrize(
    "rows,cols,tex_size",
    [
        (50, 50, 32),
        (100, 100, 64),
        (80, 80, 32),
        (60, 60, 64),
    ],
)
def test_uv_texture_sample_equivalence(rows, cols, tex_size):
    assert py_func(rows, cols, tex_size) == cy_func(rows, cols, tex_size)
