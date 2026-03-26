"""Test relativistic_energy equivalence."""

import pytest

from cnake_charmer.cy.physics.relativistic_energy import relativistic_energy as cy_func
from cnake_charmer.py.physics.relativistic_energy import relativistic_energy as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_relativistic_energy_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < abs(py_result) * 1e-9 + 1e-6, (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
