"""Test goertzel equivalence."""

import pytest

from cnake_charmer.cy.dsp.goertzel import goertzel as cy_goertzel
from cnake_charmer.py.dsp.goertzel import goertzel as py_goertzel


@pytest.mark.parametrize("n", [1000, 5000, 10000, 50000])
def test_goertzel_equivalence(n):
    py_result = py_goertzel(n)
    cy_result = cy_goertzel(n)
    assert abs(py_result - cy_result) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
