"""Test prime_sum_trial_division equivalence."""

import pytest

from cnake_data.cy.algorithms.prime_sum_trial_division import prime_sum_trial_division as cy_func
from cnake_data.py.algorithms.prime_sum_trial_division import prime_sum_trial_division as py_func


@pytest.mark.parametrize(
    "limit,start,step",
    [
        (100, 2, 1),
        (1000, 3, 2),
        (5000, 5, 1),
        (10000, 11, 2),
    ],
)
def test_prime_sum_trial_division_equivalence(limit, start, step):
    assert py_func(limit, start, step) == cy_func(limit, start, step)
