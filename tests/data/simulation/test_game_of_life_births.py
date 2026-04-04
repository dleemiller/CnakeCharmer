"""Test game_of_life_births equivalence."""

import pytest

from cnake_data.cy.simulation.game_of_life_births import game_of_life_births as cy_func
from cnake_data.py.simulation.game_of_life_births import game_of_life_births as py_func


@pytest.mark.parametrize("n", [1, 5, 10, 20])
def test_game_of_life_births_equivalence(n):
    assert py_func(n) == cy_func(n)
