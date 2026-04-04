"""Test prime_sieve equivalence."""

import pytest

from cnake_data.cy.math_problems.prime_sieve import prime_sieve as cy_func
from cnake_data.py.math_problems.prime_sieve import prime_sieve as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_prime_sieve_equivalence(n):
    assert py_func(n) == cy_func(n)
