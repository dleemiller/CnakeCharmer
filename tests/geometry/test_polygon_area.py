"""Test polygon_area equivalence."""

import pytest

from cnake_charmer.cy.geometry.polygon_area import polygon_area as cy_func
from cnake_charmer.py.geometry.polygon_area import polygon_area as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_polygon_area_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
