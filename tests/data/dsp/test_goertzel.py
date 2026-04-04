"""Test goertzel equivalence."""

import pytest

from cnake_data.cy.dsp.goertzel import goertzel as cy_goertzel
from cnake_data.py.dsp.goertzel import goertzel as py_goertzel


@pytest.mark.parametrize("n", [1000, 5000, 10000, 50000])
def test_goertzel_equivalence(n):
    py_result = py_goertzel(n)
    cy_result = cy_goertzel(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
