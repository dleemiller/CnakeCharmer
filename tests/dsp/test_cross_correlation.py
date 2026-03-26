"""Test cross_correlation equivalence."""

import pytest

from cnake_charmer.cy.dsp.cross_correlation import cross_correlation as cy_cross_correlation
from cnake_charmer.py.dsp.cross_correlation import cross_correlation as py_cross_correlation


@pytest.mark.parametrize("n", [100, 500, 1000, 2000])
def test_cross_correlation_equivalence(n):
    py_result = py_cross_correlation(n)
    cy_result = cy_cross_correlation(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
