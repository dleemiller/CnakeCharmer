"""Test particle_momentum equivalence."""

import pytest

from cnake_data.cy.physics.particle_momentum import particle_momentum_sum as cy_momentum
from cnake_data.py.physics.particle_momentum import particle_momentum_sum as py_momentum


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_particle_momentum_sum_equivalence(n):
    assert abs(py_momentum(n) - cy_momentum(n)) < 1e-4
