"""Test cpdef_gcd_sum equivalence."""

import pytest

from cnake_charmer.cy.math_problems.cpdef_gcd_sum import cpdef_gcd_sum as cy_func
from cnake_charmer.py.math_problems.cpdef_gcd_sum import cpdef_gcd_sum as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_cpdef_gcd_sum_equivalence(n):
    assert py_func(n) == cy_func(n)
