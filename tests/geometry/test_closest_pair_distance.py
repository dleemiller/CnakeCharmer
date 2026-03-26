"""Test closest_pair_distance equivalence."""

import pytest

from cnake_charmer.cy.geometry.closest_pair_distance import closest_pair_distance as cy_func
from cnake_charmer.py.geometry.closest_pair_distance import closest_pair_distance as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_closest_pair_distance_equivalence(n):
    assert abs(py_func(n) - cy_func(n)) < 1e-6
