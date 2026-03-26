"""Test voronoi_nearest equivalence."""

import pytest

from cnake_charmer.cy.geometry.voronoi_nearest import voronoi_nearest as cy_func
from cnake_charmer.py.geometry.voronoi_nearest import voronoi_nearest as py_func


@pytest.mark.parametrize("n", [10, 100, 500])
def test_voronoi_nearest_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
