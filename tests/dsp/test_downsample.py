"""Test downsample equivalence."""

import pytest

from cnake_charmer.cy.dsp.downsample import downsample as cy_downsample
from cnake_charmer.py.dsp.downsample import downsample as py_downsample


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_downsample_equivalence(n):
    py_result = py_downsample(n)
    cy_result = cy_downsample(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
