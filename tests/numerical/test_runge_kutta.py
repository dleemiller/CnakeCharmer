"""Test runge_kutta equivalence."""

import pytest

from cnake_charmer.cy.numerical.runge_kutta import runge_kutta as cy_runge_kutta
from cnake_charmer.py.numerical.runge_kutta import runge_kutta as py_runge_kutta


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_runge_kutta_equivalence(n):
    py_result = py_runge_kutta(n)
    cy_result = cy_runge_kutta(n)
    assert abs(py_result - cy_result) < 1e-9, f"Mismatch: py={py_result}, cy={cy_result}"
