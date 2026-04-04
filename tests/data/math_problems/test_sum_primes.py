"""Test sum_primes equivalence."""

import pytest

from cnake_data.cy.math_problems.sum_primes import sum_primes as cy_func
from cnake_data.py.math_problems.sum_primes import sum_primes as py_func


@pytest.mark.parametrize("n", [2, 10, 100, 500, 1000, 10000])
def test_sum_primes_equivalence(n):
    assert py_func(n) == cy_func(n)
