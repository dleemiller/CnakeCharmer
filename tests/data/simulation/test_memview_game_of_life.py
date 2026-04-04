"""Test Game of Life memoryview equivalence."""

import pytest

from cnake_data.cy.simulation.memview_game_of_life import (
    memview_game_of_life as cy_memview_game_of_life,
)
from cnake_data.py.simulation.memview_game_of_life import (
    memview_game_of_life as py_memview_game_of_life,
)


@pytest.mark.parametrize("n", [10, 30, 50, 200])
def test_memview_game_of_life_equivalence(n):
    py_result = py_memview_game_of_life(n)
    cy_result = cy_memview_game_of_life(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
