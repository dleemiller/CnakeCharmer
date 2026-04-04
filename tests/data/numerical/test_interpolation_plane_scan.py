"""Test interpolation_plane_scan equivalence."""

import pytest

from cnake_data.cy.numerical.interpolation_plane_scan import (
    interpolation_plane_scan as cy_func,
)
from cnake_data.py.numerical.interpolation_plane_scan import (
    interpolation_plane_scan as py_func,
)


@pytest.mark.parametrize(
    "a,b,c,x0,x1,y0,y1,passes,blend",
    [
        (0.1, -0.02, 1.5, -8, 8, -6, 6, 2, 0.7),
        (0.17, -0.05, 1.3, -16, 16, -12, 12, 3, 0.73),
        (-0.09, 0.12, -0.4, -10, 14, -9, 11, 4, 0.65),
    ],
)
def test_interpolation_plane_scan_equivalence(a, b, c, x0, x1, y0, y1, passes, blend):
    py_result = py_func(a, b, c, x0, x1, y0, y1, passes, blend)
    cy_result = cy_func(a, b, c, x0, x1, y0, y1, passes, blend)
    for p, c_ in zip(py_result, cy_result, strict=False):
        assert abs(p - c_) / max(abs(p), 1.0) < 1e-9
