"""Test envelope_detection equivalence."""

import pytest

from cnake_data.cy.dsp.envelope_detection import envelope_detection as cy_envelope_detection
from cnake_data.py.dsp.envelope_detection import envelope_detection as py_envelope_detection


@pytest.mark.parametrize("n", [50, 100, 200, 500])
def test_envelope_detection_equivalence(n):
    py_result = py_envelope_detection(n)
    cy_result = cy_envelope_detection(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
