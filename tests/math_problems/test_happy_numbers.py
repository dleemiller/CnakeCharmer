"""Test happy_numbers equivalence."""

import pytest

from cnake_charmer.cy.math_problems.happy_numbers import happy_numbers as cy_func
from cnake_charmer.py.math_problems.happy_numbers import happy_numbers as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_happy_numbers_equivalence(n):
    assert py_func(n) == cy_func(n)
