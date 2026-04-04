"""Test capacitor_discharge equivalence."""

import pytest

from cnake_data.cy.physics.capacitor_discharge import capacitor_discharge as cy_func
from cnake_data.py.physics.capacitor_discharge import capacitor_discharge as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_capacitor_discharge_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1e-12) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
