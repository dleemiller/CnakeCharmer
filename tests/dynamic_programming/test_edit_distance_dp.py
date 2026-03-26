"""Test edit_distance_dp equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.edit_distance_dp import edit_distance_dp as cy_func
from cnake_charmer.py.dynamic_programming.edit_distance_dp import edit_distance_dp as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_edit_distance_dp_equivalence(n):
    assert py_func(n) == cy_func(n)
