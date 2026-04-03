"""Test triangle_angles_3d equivalence."""

import pytest

from cnake_charmer.cy.geometry.triangle_angles_3d import triangle_angles_3d as cy_func
from cnake_charmer.py.geometry.triangle_angles_3d import triangle_angles_3d as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_triangle_angles_3d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
