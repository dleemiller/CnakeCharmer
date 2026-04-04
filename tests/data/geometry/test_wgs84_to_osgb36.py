"""Test wgs84_to_osgb36 equivalence."""

import pytest

from cnake_data.cy.geometry.wgs84_to_osgb36 import wgs84_to_osgb36 as cy_wgs84_to_osgb36
from cnake_data.py.geometry.wgs84_to_osgb36 import wgs84_to_osgb36 as py_wgs84_to_osgb36


@pytest.mark.parametrize("n", [1, 10, 50, 100])
def test_wgs84_to_osgb36_equivalence(n):
    assert py_wgs84_to_osgb36(n) == cy_wgs84_to_osgb36(n)
