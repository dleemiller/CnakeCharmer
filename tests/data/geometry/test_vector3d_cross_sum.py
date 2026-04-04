"""Test vector3d_cross_sum equivalence."""

import pytest

from cnake_data.cy.geometry.vector3d_cross_sum import vector3d_cross_sum as cy_func
from cnake_data.py.geometry.vector3d_cross_sum import vector3d_cross_sum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_vector3d_cross_sum_equivalence(n):
    py_x, py_y, py_z = py_func(n)
    cy_x, cy_y, cy_z = cy_func(n)
    tol_x = max(1e-6, abs(py_x) * 1e-9)
    tol_y = max(1e-6, abs(py_y) * 1e-9)
    tol_z = max(1e-6, abs(py_z) * 1e-9)
    assert abs(py_x - cy_x) < tol_x, f"X mismatch: py={py_x}, cy={cy_x}"
    assert abs(py_y - cy_y) < tol_y, f"Y mismatch: py={py_y}, cy={cy_y}"
    assert abs(py_z - cy_z) < tol_z, f"Z mismatch: py={py_z}, cy={cy_z}"
