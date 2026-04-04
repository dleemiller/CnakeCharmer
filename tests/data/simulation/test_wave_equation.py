"""Test wave_equation equivalence."""

import pytest

from cnake_data.cy.simulation.wave_equation import wave_equation as cy_func
from cnake_data.py.simulation.wave_equation import wave_equation as py_func


@pytest.mark.parametrize("n", [10, 20, 50, 80])
def test_wave_equation_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
