"""Test polygon_area_centroid equivalence."""

import pytest

from cnake_data.cy.geometry.polygon_area_centroid import polygon_area_centroid as cy_func
from cnake_data.py.geometry.polygon_area_centroid import polygon_area_centroid as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_polygon_area_centroid_equivalence(n):
    py_area, py_cx, py_cy = py_func(n)
    cy_area, cy_cx, cy_cy = cy_func(n)
    tol = max(1e-3, abs(py_area) * 1e-9)
    assert abs(py_area - cy_area) < tol, f"Area mismatch: py={py_area}, cy={cy_area}"
    assert abs(py_cx - cy_cx) < 1e-3, f"Centroid X mismatch: py={py_cx}, cy={cy_cx}"
    assert abs(py_cy - cy_cy) < 1e-3, f"Centroid Y mismatch: py={py_cy}, cy={cy_cy}"
