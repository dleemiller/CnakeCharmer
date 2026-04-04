"""Test sieve_of_eratosthenes equivalence."""

import pytest

from cnake_charmer.cy.algorithms.sieve_of_eratosthenes import sieve_of_eratosthenes as cy_sieve
from cnake_charmer.py.algorithms.sieve_of_eratosthenes import sieve_of_eratosthenes as py_sieve


@pytest.mark.parametrize("n", [100, 10000, 500000, 5000000])
def test_sieve_of_eratosthenes_equivalence(n):
    py_result = py_sieve(n)
    cy_result = cy_sieve(n)
    assert py_result == cy_result, f"Mismatch at n={n}: py={py_result}, cy={cy_result}"
