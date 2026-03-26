"""Test polynomial_eval equivalence."""

import pytest

from cnake_charmer.cy.numerical.polynomial_eval import polynomial_eval as cy_polynomial_eval
from cnake_charmer.py.numerical.polynomial_eval import polynomial_eval as py_polynomial_eval


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_polynomial_eval_equivalence(n):
    py_result = py_polynomial_eval(n)
    cy_result = cy_polynomial_eval(n)
    # Polynomial evaluation can accumulate floating point errors
    rel_tol = abs(py_result) * 1e-6 if py_result != 0 else 1e-6
    assert abs(py_result - cy_result) < rel_tol, f"Mismatch: py={py_result}, cy={cy_result}"
