"""Test convex_hull_perimeter equivalence."""

import pytest

from cnake_data.cy.geometry.convex_hull_perimeter import convex_hull_perimeter as cy_func
from cnake_data.py.geometry.convex_hull_perimeter import convex_hull_perimeter as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_convex_hull_perimeter_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert isinstance(py_result, tuple)
    assert isinstance(cy_result, tuple)
    # Check perimeter (float)
    assert abs(py_result[0] - cy_result[0]) / max(abs(py_result[0]), 1.0) < 1e-6, (
        f"Perimeter mismatch: py={py_result[0]}, cy={cy_result[0]}"
    )
    # Check hull vertex count (int)
    assert py_result[1] == cy_result[1], f"Hull size mismatch: py={py_result[1]}, cy={cy_result[1]}"
