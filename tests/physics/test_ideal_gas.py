"""Test ideal_gas equivalence."""

import pytest

from cnake_charmer.cy.physics.ideal_gas import ideal_gas as cy_func
from cnake_charmer.py.physics.ideal_gas import ideal_gas as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_ideal_gas_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < abs(py_result) * 1e-9 + 1e-6, (
        f"Mismatch: py={py_result}, cy={cy_result}"
    )
