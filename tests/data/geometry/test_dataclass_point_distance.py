"""Test dataclass_point_distance equivalence."""

import pytest

from cnake_data.cy.geometry.dataclass_point_distance import dataclass_point_distance as cy_func
from cnake_data.py.geometry.dataclass_point_distance import dataclass_point_distance as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 3000])
def test_dataclass_point_distance_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-4
