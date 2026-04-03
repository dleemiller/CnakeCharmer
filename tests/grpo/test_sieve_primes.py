"""Test sieve_primes."""

import pytest

from cnake_charmer.py.grpo.sieve_primes import sieve_primes


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_sieve_primes(n):
    result = sieve_primes(n)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result == sieve_primes(n)
