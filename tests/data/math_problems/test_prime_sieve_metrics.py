"""Test prime_sieve_metrics equivalence."""

import pytest

from cnake_data.cy.math_problems.prime_sieve_metrics import prime_sieve_metrics as cy_func
from cnake_data.py.math_problems.prime_sieve_metrics import prime_sieve_metrics as py_func


@pytest.mark.parametrize(
    "limit,window,mod_base",
    [
        (100, 16, 10007),
        (1000, 32, 1000003),
        (5000, 64, 999983),
    ],
)
def test_prime_sieve_metrics_equivalence(limit, window, mod_base):
    assert py_func(limit, window, mod_base) == cy_func(limit, window, mod_base)
