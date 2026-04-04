"""Test png_filters equivalence."""

import pytest

from cnake_data.cy.compression.png_filters import png_filters as cy_func
from cnake_data.py.compression.png_filters import png_filters as py_func


@pytest.mark.parametrize("width,height,fu", [(10, 10, 1), (50, 50, 3), (100, 100, 4)])
def test_png_filters_equivalence(width, height, fu):
    assert py_func(width, height, fu) == cy_func(width, height, fu)
