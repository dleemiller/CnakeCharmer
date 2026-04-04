"""Test game_of_life equivalence."""

import pytest

from cnake_data.cy.simulation.game_of_life import game_of_life as cy_func
from cnake_data.py.simulation.game_of_life import game_of_life as py_func


@pytest.mark.parametrize("n", [10, 20, 50, 100])
def test_game_of_life_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
