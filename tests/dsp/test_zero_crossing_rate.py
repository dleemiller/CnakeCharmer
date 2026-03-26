"""Test zero_crossing_rate equivalence."""

import pytest

from cnake_charmer.cy.dsp.zero_crossing_rate import zero_crossing_rate as cy_zero_crossing_rate
from cnake_charmer.py.dsp.zero_crossing_rate import zero_crossing_rate as py_zero_crossing_rate


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_zero_crossing_rate_equivalence(n):
    py_result = py_zero_crossing_rate(n)
    cy_result = cy_zero_crossing_rate(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
