"""Test polygon_winding equivalence."""

import pytest

from cnake_charmer.cy.geometry.polygon_winding import polygon_winding as cy_func
from cnake_charmer.py.geometry.polygon_winding import polygon_winding as py_func


@pytest.mark.parametrize("n", [100, 500, 1000, 3000])
def test_polygon_winding_equivalence(n):
    assert py_func(n) == cy_func(n)
