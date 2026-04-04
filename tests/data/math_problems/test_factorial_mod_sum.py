"""Test factorial_mod_sum equivalence."""

import pytest

from cnake_data.cy.math_problems.factorial_mod_sum import factorial_mod_sum as cy_func
from cnake_data.py.math_problems.factorial_mod_sum import factorial_mod_sum as py_func


@pytest.mark.parametrize(
    "limit,mod",
    [(10, 1_000_000_007), (1000, 1_000_000_007), (5000, 998244353)],
)
def test_factorial_mod_sum_equivalence(limit, mod):
    assert py_func(limit, mod) == cy_func(limit, mod)
