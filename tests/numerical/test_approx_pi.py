"""Test Monte Carlo pi approximation equivalence."""

import pytest

from cnake_charmer.cy.numerical.approx_pi import approx_pi as cy_approx_pi
from cnake_charmer.py.numerical.approx_pi import approx_pi as py_approx_pi


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_approx_pi_equivalence(n):
    py_result = py_approx_pi(n)
    cy_result = cy_approx_pi(n)
    assert abs(py_result - cy_result) < 1e-9, f"Mismatch: py={py_result}, cy={cy_result}"
