"""Test harmonic_oscillator equivalence."""

import pytest

from cnake_data.cy.physics.harmonic_oscillator import harmonic_oscillator as cy_func
from cnake_data.py.physics.harmonic_oscillator import harmonic_oscillator as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_harmonic_oscillator_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < abs(py_result) * 1e-6 + 1e-6, (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
