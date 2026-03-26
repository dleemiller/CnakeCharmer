"""Test fir_filter equivalence."""

import pytest

from cnake_charmer.cy.dsp.fir_filter import fir_filter as cy_fir_filter
from cnake_charmer.py.dsp.fir_filter import fir_filter as py_fir_filter


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_fir_filter_equivalence(n):
    py_result = py_fir_filter(n)
    cy_result = cy_fir_filter(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
