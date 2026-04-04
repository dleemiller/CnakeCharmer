"""Test particle_pair_momentum_class equivalence."""

import pytest

from cnake_data.cy.simulation.particle_pair_momentum_class import (
    particle_pair_momentum_class as cy_func,
)
from cnake_data.py.simulation.particle_pair_momentum_class import (
    particle_pair_momentum_class as py_func,
)


@pytest.mark.parametrize("n,rounds,coupling", [(20, 60, 0.05), (24, 70, 0.07), (18, 80, 0.09)])
def test_particle_pair_momentum_class_equivalence(n, rounds, coupling):
    py_result = py_func(n, rounds, coupling)
    cy_result = cy_func(n, rounds, coupling)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-8
