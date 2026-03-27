"""Test autocorrelation equivalence."""

import pytest

from cnake_charmer.cy.dsp.autocorrelation import autocorrelation as cy_autocorrelation
from cnake_charmer.py.dsp.autocorrelation import autocorrelation as py_autocorrelation


@pytest.mark.parametrize("n", [200, 500, 1000, 5000])
def test_autocorrelation_equivalence(n):
    py_result = py_autocorrelation(n)
    cy_result = cy_autocorrelation(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
