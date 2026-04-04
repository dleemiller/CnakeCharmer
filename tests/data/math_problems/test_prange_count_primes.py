"""Test prange_count_primes equivalence."""

import pytest

from cnake_data.cy.math_problems.prange_count_primes import (
    prange_count_primes as cy_func,
)
from cnake_data.py.math_problems.prange_count_primes import (
    prange_count_primes as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_prange_count_primes_equivalence(n):
    assert py_func(n) == cy_func(n)
