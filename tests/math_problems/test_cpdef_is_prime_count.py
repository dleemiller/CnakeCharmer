"""Test cpdef_is_prime_count equivalence."""

import pytest

from cnake_charmer.cy.math_problems.cpdef_is_prime_count import (
    cpdef_is_prime_count as cy_func,
)
from cnake_charmer.py.math_problems.cpdef_is_prime_count import (
    cpdef_is_prime_count as py_func,
)


@pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
def test_cpdef_is_prime_count_equivalence(n):
    assert py_func(n) == cy_func(n)
