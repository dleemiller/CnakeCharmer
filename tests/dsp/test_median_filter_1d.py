"""Test median_filter_1d equivalence."""

import pytest

from cnake_charmer.cy.dsp.median_filter_1d import median_filter_1d as cy_func
from cnake_charmer.py.dsp.median_filter_1d import median_filter_1d as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_median_filter_1d_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
