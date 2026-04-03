"""Test euclidean_gcd_sum equivalence."""

import pytest

from cnake_charmer.cy.math_problems.euclidean_gcd_sum import euclidean_gcd_sum as cy_func
from cnake_charmer.py.math_problems.euclidean_gcd_sum import euclidean_gcd_sum as py_func


@pytest.mark.parametrize("seed_a,seed_b,count", [(1, 2, 10), (6789, 9876, 1000), (9, 11, 5000)])
def test_euclidean_gcd_sum_equivalence(seed_a, seed_b, count):
    assert py_func(seed_a, seed_b, count) == cy_func(seed_a, seed_b, count)
