"""Test matrix2x2_power equivalence."""

import pytest

from cnake_charmer.cy.numerical.matrix2x2_power import matrix2x2_power as cy_func
from cnake_charmer.py.numerical.matrix2x2_power import matrix2x2_power as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_matrix2x2_power_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    tol = max(1e-6, abs(py_result) * 1e-9)
    assert abs(py_result - cy_result) < tol, f"Mismatch: py={py_result}, cy={cy_result}"
