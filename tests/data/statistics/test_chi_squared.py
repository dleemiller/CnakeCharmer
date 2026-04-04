"""Test chi_squared equivalence."""

import pytest

from cnake_data.cy.statistics.chi_squared import chi_squared as cy_chi_squared
from cnake_data.py.statistics.chi_squared import chi_squared as py_chi_squared


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_chi_squared_equivalence(n):
    py_result = py_chi_squared(n)
    cy_result = cy_chi_squared(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
