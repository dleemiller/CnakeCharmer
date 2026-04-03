"""Test laguerre polynomial equivalence."""

import pytest

from cnake_charmer.cy.math_problems.laguerre_polynomial import (
    laguerre_polynomial as cy_laguerre_polynomial,
)
from cnake_charmer.py.math_problems.laguerre_polynomial import (
    laguerre_polynomial as py_laguerre_polynomial,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_laguerre_polynomial_equivalence(n):
    py_result = py_laguerre_polynomial(n)
    cy_result = cy_laguerre_polynomial(n)
    assert len(py_result) == len(cy_result) == 2
    for py_val, cy_val in zip(py_result, cy_result, strict=False):
        assert abs(py_val - cy_val) / max(abs(py_val), 1.0) < 1e-4
