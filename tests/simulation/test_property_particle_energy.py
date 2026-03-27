"""Test property_particle_energy equivalence."""

import pytest

from cnake_charmer.cy.simulation.property_particle_energy import (
    property_particle_energy as cy_func,
)
from cnake_charmer.py.simulation.property_particle_energy import (
    property_particle_energy as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500, 2000])
def test_property_particle_energy_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-6, abs(py_result) * 1e-9), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
