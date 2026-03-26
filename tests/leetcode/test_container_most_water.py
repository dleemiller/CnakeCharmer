"""Test container_most_water equivalence."""

import pytest

from cnake_charmer.cy.leetcode.container_most_water import container_most_water as cy_func
from cnake_charmer.py.leetcode.container_most_water import container_most_water as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_container_most_water_equivalence(n):
    assert py_func(n) == cy_func(n)
