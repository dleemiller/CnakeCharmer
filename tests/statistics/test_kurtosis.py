"""Test kurtosis equivalence."""

import pytest

from cnake_charmer.cy.statistics.kurtosis import kurtosis as cy_func
from cnake_charmer.py.statistics.kurtosis import kurtosis as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_kurtosis_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
