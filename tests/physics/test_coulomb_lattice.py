"""Test coulomb_lattice equivalence."""

import pytest

from cnake_charmer.cy.physics.coulomb_lattice import coulomb_lattice as cy_coulomb_lattice
from cnake_charmer.py.physics.coulomb_lattice import coulomb_lattice as py_coulomb_lattice


@pytest.mark.parametrize("n", [10, 100, 500, 2000])
def test_coulomb_lattice_equivalence(n):
    py_result = py_coulomb_lattice(n)
    cy_result = cy_coulomb_lattice(n)
    assert py_result == cy_result, f"Mismatch at n={n}: py={py_result}, cy={cy_result}"
