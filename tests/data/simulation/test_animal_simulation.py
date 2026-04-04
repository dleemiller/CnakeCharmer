"""Test animal_simulation equivalence."""

import pytest

from cnake_data.cy.simulation.animal_simulation import animal_simulation as cy_func
from cnake_data.py.simulation.animal_simulation import animal_simulation as py_func


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_animal_simulation_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < max(1e-3, abs(py_result) * 1e-3), (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
