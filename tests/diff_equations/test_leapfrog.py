"""Test leapfrog equivalence."""

import pytest

from cnake_charmer.cy.diff_equations.leapfrog import leapfrog as cy_func
from cnake_charmer.py.diff_equations.leapfrog import leapfrog as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_leapfrog_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
