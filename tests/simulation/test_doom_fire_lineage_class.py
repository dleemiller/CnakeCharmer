"""Test doom_fire_lineage_class equivalence."""

import pytest

from cnake_charmer.cy.simulation.doom_fire_lineage_class import doom_fire_lineage_class as cy_func
from cnake_charmer.py.simulation.doom_fire_lineage_class import doom_fire_lineage_class as py_func


@pytest.mark.parametrize(
    "width,height,steps,cooling,seed", [(28, 24, 30, 2, 3), (36, 28, 45, 3, 9), (44, 32, 35, 4, 19)]
)
def test_doom_fire_lineage_class_equivalence(width, height, steps, cooling, seed):
    assert py_func(width, height, steps, cooling, seed) == cy_func(
        width, height, steps, cooling, seed
    )
