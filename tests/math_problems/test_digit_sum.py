"""Test digit_sum equivalence."""

import pytest

from cnake_charmer.cy.math_problems.digit_sum import digit_sum as cy_func
from cnake_charmer.py.math_problems.digit_sum import digit_sum as py_func


@pytest.mark.parametrize("n", [1, 10, 100, 1000, 10000])
def test_digit_sum_equivalence(n):
    assert py_func(n) == cy_func(n)
