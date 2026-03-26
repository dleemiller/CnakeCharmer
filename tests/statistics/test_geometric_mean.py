"""Test geometric_mean equivalence."""

import pytest

from cnake_charmer.cy.statistics.geometric_mean import geometric_mean as cy_func
from cnake_charmer.py.statistics.geometric_mean import geometric_mean as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_geometric_mean_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
