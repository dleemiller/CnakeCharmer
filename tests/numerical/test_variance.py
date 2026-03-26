"""Test variance equivalence."""

import pytest

from cnake_charmer.cy.numerical.variance import variance as cy_variance
from cnake_charmer.py.numerical.variance import variance as py_variance


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_variance_equivalence(n):
    py_result = py_variance(n)
    cy_result = cy_variance(n)
    assert abs(py_result - cy_result) < 1e-3, f"Mismatch: py={py_result}, cy={cy_result}"
