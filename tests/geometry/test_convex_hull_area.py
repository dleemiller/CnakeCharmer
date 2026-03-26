"""Test convex_hull_area equivalence."""

import pytest

from cnake_charmer.cy.geometry.convex_hull_area import convex_hull_area as cy_func
from cnake_charmer.py.geometry.convex_hull_area import convex_hull_area as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_convex_hull_area_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
