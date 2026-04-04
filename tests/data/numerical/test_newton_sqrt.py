"""Test newton_sqrt equivalence."""

import pytest

from cnake_data.cy.numerical.newton_sqrt import newton_sqrt as cy_newton_sqrt
from cnake_data.py.numerical.newton_sqrt import newton_sqrt as py_newton_sqrt


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_newton_sqrt_equivalence(n):
    py_result = py_newton_sqrt(n)
    cy_result = cy_newton_sqrt(n)
    assert abs(py_result - cy_result) < 1e-6, f"Mismatch: py={py_result}, cy={cy_result}"
