"""Test monte_carlo_integration equivalence."""

import pytest

from cnake_charmer.cy.simulation.monte_carlo_integration import monte_carlo_integration as cy_func
from cnake_charmer.py.simulation.monte_carlo_integration import monte_carlo_integration as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_monte_carlo_integration_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1e-12) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
