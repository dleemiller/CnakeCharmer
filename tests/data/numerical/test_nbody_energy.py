"""Test nbody_energy equivalence."""

import pytest

from cnake_data.cy.numerical.nbody_energy import nbody_energy as cy_nbody_energy
from cnake_data.py.numerical.nbody_energy import nbody_energy as py_nbody_energy


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_nbody_energy_equivalence(n):
    py_result = py_nbody_energy(n)
    cy_result = cy_nbody_energy(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
