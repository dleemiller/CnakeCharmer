"""Test count_primes_trial equivalence."""

import pytest

from cnake_data.cy.math_problems.count_primes_trial import count_primes_trial as cy_func
from cnake_data.py.math_problems.count_primes_trial import count_primes_trial as py_func


@pytest.mark.parametrize("limit,return_last", [(10, True), (500, True), (1000, False)])
def test_count_primes_trial_equivalence(limit, return_last):
    assert py_func(limit, return_last) == cy_func(limit, return_last)
