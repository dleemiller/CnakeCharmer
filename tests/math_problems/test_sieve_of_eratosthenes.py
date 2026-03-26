"""Test sieve of Eratosthenes equivalence."""

import pytest

from cnake_charmer.cy.math_problems.sieve_of_eratosthenes import (
    sieve_of_eratosthenes as cy_sieve_of_eratosthenes,
)
from cnake_charmer.py.math_problems.sieve_of_eratosthenes import (
    sieve_of_eratosthenes as py_sieve_of_eratosthenes,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_sieve_of_eratosthenes_equivalence(n):
    assert py_sieve_of_eratosthenes(n) == cy_sieve_of_eratosthenes(n)
