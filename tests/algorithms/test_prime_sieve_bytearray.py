"""Test prime_sieve_bytearray equivalence."""

import pytest

from cnake_charmer.cy.algorithms.prime_sieve_bytearray import (
    prime_sieve_bytearray as cy_prime_sieve_bytearray,
)
from cnake_charmer.py.algorithms.prime_sieve_bytearray import (
    prime_sieve_bytearray as py_prime_sieve_bytearray,
)


@pytest.mark.parametrize("limit", [10, 100, 1000, 10000])
def test_prime_sieve_bytearray_equivalence(limit):
    assert py_prime_sieve_bytearray(limit) == cy_prime_sieve_bytearray(limit)
