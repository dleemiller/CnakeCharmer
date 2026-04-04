"""Test point_in_polygon equivalence."""

import pytest

from cnake_data.cy.geometry.point_in_polygon import point_in_polygon as cy_func
from cnake_data.py.geometry.point_in_polygon import point_in_polygon as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_point_in_polygon_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
