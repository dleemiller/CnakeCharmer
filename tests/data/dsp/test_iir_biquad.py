"""Test iir_biquad equivalence."""

import pytest

from cnake_data.cy.dsp.iir_biquad import iir_biquad as cy_iir_biquad
from cnake_data.py.dsp.iir_biquad import iir_biquad as py_iir_biquad


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_iir_biquad_equivalence(n):
    py_result = py_iir_biquad(n)
    cy_result = cy_iir_biquad(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
