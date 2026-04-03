"""Test integer_sqrt_newton equivalence."""

import pytest

from cnake_charmer.cy.math_problems.integer_sqrt_newton import integer_sqrt_newton as cy_func
from cnake_charmer.py.math_problems.integer_sqrt_newton import integer_sqrt_newton as py_func


@pytest.mark.parametrize("start,stop", [(1, 10), (1, 100), (100, 1000), (0, 2000)])
def test_integer_sqrt_newton_equivalence(start, stop):
    assert py_func(start, stop) == cy_func(start, stop)
