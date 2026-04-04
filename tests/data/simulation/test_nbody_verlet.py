"""Test nbody_verlet equivalence."""

import pytest

from cnake_data.cy.simulation.nbody_verlet import nbody_verlet as cy_func
from cnake_data.py.simulation.nbody_verlet import nbody_verlet as py_func


@pytest.mark.parametrize("n", [5, 10, 50, 100])
def test_nbody_verlet_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
