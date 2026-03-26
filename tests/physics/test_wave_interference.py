"""Test wave_interference equivalence."""

import pytest

from cnake_charmer.cy.physics.wave_interference import wave_interference as cy_func
from cnake_charmer.py.physics.wave_interference import wave_interference as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_wave_interference_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < abs(py_result) * 1e-6 + 1e-6, (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
