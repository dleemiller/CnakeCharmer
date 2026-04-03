"""Test is_prime_while equivalence."""

import pytest

from cnake_charmer.cy.math_problems.is_prime_while import (
    count_primes_while as cy_count_primes_while,
)
from cnake_charmer.py.math_problems.is_prime_while import (
    count_primes_while as py_count_primes_while,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_count_primes_while_equivalence(n):
    assert py_count_primes_while(n) == cy_count_primes_while(n)
