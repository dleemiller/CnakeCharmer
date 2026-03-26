"""Test hilbert_envelope equivalence."""

import pytest

from cnake_charmer.cy.dsp.hilbert_envelope import hilbert_envelope as cy_func
from cnake_charmer.py.dsp.hilbert_envelope import hilbert_envelope as py_func


@pytest.mark.parametrize("n", [50, 100, 200, 500])
def test_hilbert_envelope_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
