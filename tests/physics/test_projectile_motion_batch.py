"""Test projectile_motion_batch equivalence."""

import pytest

from cnake_charmer.cy.physics.projectile_motion_batch import projectile_motion_batch as cy_func
from cnake_charmer.py.physics.projectile_motion_batch import projectile_motion_batch as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_projectile_motion_batch_equivalence(n):
    assert py_func(n) == cy_func(n)
