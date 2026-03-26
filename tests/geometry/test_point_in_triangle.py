"""Test point_in_triangle equivalence."""

import pytest

from cnake_charmer.cy.geometry.point_in_triangle import point_in_triangle as cy_func
from cnake_charmer.py.geometry.point_in_triangle import point_in_triangle as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_point_in_triangle_equivalence(n):
    assert py_func(n) == cy_func(n)
