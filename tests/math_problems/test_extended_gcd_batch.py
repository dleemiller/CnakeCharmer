"""Test extended_gcd_batch equivalence."""

import pytest

from cnake_charmer.cy.math_problems.extended_gcd_batch import extended_gcd_batch as cy_func
from cnake_charmer.py.math_problems.extended_gcd_batch import extended_gcd_batch as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_extended_gcd_batch_equivalence(n):
    assert py_func(n) == cy_func(n)
