"""Test life_board_steps_class equivalence."""

import pytest

from cnake_data.cy.simulation.life_board_steps_class import life_board_steps_class as cy_func
from cnake_data.py.simulation.life_board_steps_class import life_board_steps_class as py_func


@pytest.mark.parametrize(
    "width,height,steps,seed", [(18, 14, 10, 5), (24, 20, 16, 11), (28, 22, 12, 17)]
)
def test_life_board_steps_class_equivalence(width, height, steps, seed):
    assert py_func(width, height, steps, seed) == cy_func(width, height, steps, seed)
