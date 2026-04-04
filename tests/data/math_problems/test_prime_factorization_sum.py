"""Test prime_factorization_sum equivalence."""

import pytest

from cnake_data.cy.math_problems.prime_factorization_sum import (
    prime_factorization_sum as cy_func,
)
from cnake_data.py.math_problems.prime_factorization_sum import (
    prime_factorization_sum as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_prime_factorization_sum_equivalence(n):
    assert py_func(n) == cy_func(n)
