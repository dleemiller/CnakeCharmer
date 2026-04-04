"""Test doppler_shift equivalence."""

import pytest

from cnake_data.cy.physics.doppler_shift import doppler_shift as cy_func
from cnake_data.py.physics.doppler_shift import doppler_shift as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_doppler_shift_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < abs(py_result) * 1e-9 + 1e-6, (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
