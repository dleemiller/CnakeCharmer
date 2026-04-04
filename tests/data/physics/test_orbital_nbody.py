"""Test orbital_nbody equivalence."""

import pytest

from cnake_data.cy.physics.orbital_nbody import orbital_nbody as cy_func
from cnake_data.py.physics.orbital_nbody import orbital_nbody as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_orbital_nbody_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for i in range(3):
        assert abs(py_result[i] - cy_result[i]) / max(abs(py_result[i]), 1.0) < 1e-4, (
            f"Mismatch at index {i}: py={py_result[i]}, cy={cy_result[i]}"
        )
