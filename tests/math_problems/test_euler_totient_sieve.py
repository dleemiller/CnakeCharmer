"""Test Euler totient sieve equivalence."""

import pytest

from cnake_charmer.cy.math_problems.euler_totient_sieve import (
    euler_totient_sieve as cy_euler_totient_sieve,
)
from cnake_charmer.py.math_problems.euler_totient_sieve import (
    euler_totient_sieve as py_euler_totient_sieve,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_euler_totient_sieve_equivalence(n):
    assert py_euler_totient_sieve(n) == cy_euler_totient_sieve(n)
