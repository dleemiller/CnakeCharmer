"""Test blackbody_radiation equivalence."""

import pytest

from cnake_charmer.cy.physics.blackbody_radiation import blackbody_radiation as cy_func
from cnake_charmer.py.physics.blackbody_radiation import blackbody_radiation as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_blackbody_radiation_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < abs(py_result) * 1e-9 + 1e-6, (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
