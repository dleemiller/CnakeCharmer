"""Test trapezoidal integration equivalence."""

import pytest

from cnake_data.cy.numerical.trapezoidal_integration import (
    trapezoidal_integration as cy_trapezoidal_integration,
)
from cnake_data.py.numerical.trapezoidal_integration import (
    trapezoidal_integration as py_trapezoidal_integration,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_trapezoidal_integration_equivalence(n):
    py_result = py_trapezoidal_integration(n)
    cy_result = cy_trapezoidal_integration(n)
    assert abs(py_result - cy_result) < 1e-9, f"Mismatch: py={py_result}, cy={cy_result}"
