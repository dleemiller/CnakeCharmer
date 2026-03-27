"""Test bisection_root equivalence."""

import pytest

from cnake_charmer.cy.numerical.bisection_root import bisection_root as cy_bisection_root
from cnake_charmer.py.numerical.bisection_root import bisection_root as py_bisection_root


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_bisection_root_equivalence(n):
    py_result = py_bisection_root(n)
    cy_result = cy_bisection_root(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
