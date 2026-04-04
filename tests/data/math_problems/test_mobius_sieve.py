"""Test mobius_sieve equivalence."""

import pytest

from cnake_data.cy.math_problems.mobius_sieve import mobius_sieve as cy_func
from cnake_data.py.math_problems.mobius_sieve import mobius_sieve as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_mobius_sieve_equivalence(n):
    assert py_func(n) == cy_func(n)
