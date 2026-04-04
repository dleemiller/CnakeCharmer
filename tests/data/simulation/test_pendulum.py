"""Test pendulum equivalence."""

import pytest

from cnake_data.cy.simulation.pendulum import pendulum as cy_func
from cnake_data.py.simulation.pendulum import pendulum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_pendulum_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
