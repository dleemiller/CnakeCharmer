"""Test pitch_detect_amdf equivalence."""

import pytest

from cnake_charmer.cy.dsp.pitch_detect_amdf import pitch_detect_amdf as cy_func
from cnake_charmer.py.dsp.pitch_detect_amdf import pitch_detect_amdf as py_func


@pytest.mark.parametrize("n", [1000, 5000, 10000, 20000])
def test_pitch_detect_amdf_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    # First element is integer lag
    assert py_result[0] == cy_result[0], f"Lag mismatch: py={py_result[0]}, cy={cy_result[0]}"
    # Float comparisons for AMDF values
    for p, c in zip(py_result[1:], cy_result[1:], strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
