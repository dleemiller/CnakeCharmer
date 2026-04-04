"""Test orbital_mechanics equivalence."""

import pytest

from cnake_data.cy.physics.orbital_mechanics import orbital_mechanics as cy_func
from cnake_data.py.physics.orbital_mechanics import orbital_mechanics as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_orbital_mechanics_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < abs(py_result) * 1e-9, (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
