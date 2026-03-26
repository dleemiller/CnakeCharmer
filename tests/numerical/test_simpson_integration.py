"""Test simpson_integration equivalence."""

import pytest

from cnake_charmer.cy.numerical.simpson_integration import (
    simpson_integration as cy_simpson_integration,
)
from cnake_charmer.py.numerical.simpson_integration import (
    simpson_integration as py_simpson_integration,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_simpson_integration_equivalence(n):
    py_result = py_simpson_integration(n)
    cy_result = cy_simpson_integration(n)
    assert abs(py_result - cy_result) < 1e-9, f"Mismatch: py={py_result}, cy={cy_result}"
