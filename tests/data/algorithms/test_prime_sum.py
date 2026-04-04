"""Test prime_sum equivalence."""

import pytest

from cnake_data.cy.algorithms.prime_sum import prime_sum as cy_func
from cnake_data.py.algorithms.prime_sum import prime_sum as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_prime_sum_equivalence(n):
    assert py_func(n) == cy_func(n)
