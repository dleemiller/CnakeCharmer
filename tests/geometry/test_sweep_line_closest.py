"""Test sweep_line_closest equivalence."""

import pytest

from cnake_charmer.cy.geometry.sweep_line_closest import sweep_line_closest as cy_func
from cnake_charmer.py.geometry.sweep_line_closest import sweep_line_closest as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_sweep_line_closest_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
