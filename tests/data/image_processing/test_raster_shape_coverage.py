"""Test raster_shape_coverage equivalence."""

import pytest

from cnake_data.cy.image_processing.raster_shape_coverage import raster_shape_coverage as cy_func
from cnake_data.py.image_processing.raster_shape_coverage import raster_shape_coverage as py_func


@pytest.mark.parametrize("args", [(32, 24, 15, 3), (48, 36, 30, 9), (64, 40, 45, 17)])
def test_raster_shape_coverage_equivalence(args):
    assert py_func(*args) == cy_func(*args)
