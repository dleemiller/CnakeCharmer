"""Test fluid_1d equivalence."""

import pytest

from cnake_data.cy.simulation.fluid_1d import fluid_1d as cy_func
from cnake_data.py.simulation.fluid_1d import fluid_1d as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_fluid_1d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
