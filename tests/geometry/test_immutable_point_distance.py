"""Test immutable_point_distance equivalence."""

import pytest

from cnake_charmer.cy.geometry.immutable_point_distance import immutable_point_distance as cy_func
from cnake_charmer.py.geometry.immutable_point_distance import immutable_point_distance as py_func


@pytest.mark.parametrize("n", [50, 100, 500, 1000])
def test_immutable_point_distance_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-4
