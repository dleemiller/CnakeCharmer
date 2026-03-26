"""Test forest_fire equivalence."""

import pytest

from cnake_charmer.cy.simulation.forest_fire import forest_fire as cy_func
from cnake_charmer.py.simulation.forest_fire import forest_fire as py_func


@pytest.mark.parametrize("n", [10, 50, 100])
def test_forest_fire_equivalence(n):
    assert py_func(n) == cy_func(n)
