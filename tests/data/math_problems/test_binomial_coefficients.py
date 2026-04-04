"""Test binomial_coefficients equivalence."""

import pytest

from cnake_data.cy.math_problems.binomial_coefficients import binomial_coefficients as cy_func
from cnake_data.py.math_problems.binomial_coefficients import binomial_coefficients as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_binomial_coefficients_equivalence(n):
    assert py_func(n) == cy_func(n)
