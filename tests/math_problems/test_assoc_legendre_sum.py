"""Test associated Legendre sum equivalence."""

import pytest

from cnake_charmer.cy.math_problems.assoc_legendre_sum import (
    assoc_legendre_sum as cy_assoc_legendre_sum,
)
from cnake_charmer.py.math_problems.assoc_legendre_sum import (
    assoc_legendre_sum as py_assoc_legendre_sum,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_assoc_legendre_sum_equivalence(n):
    py_result = py_assoc_legendre_sum(n)
    cy_result = cy_assoc_legendre_sum(n)
    for p, c in zip(py_result, cy_result, strict=False):
        assert abs(p - c) / max(abs(p), 1.0) < 1e-4, f"Mismatch: py={py_result}, cy={cy_result}"
