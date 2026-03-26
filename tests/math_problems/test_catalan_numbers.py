"""Test catalan_numbers equivalence."""

import pytest

from cnake_charmer.cy.math_problems.catalan_numbers import catalan_numbers as cy_func
from cnake_charmer.py.math_problems.catalan_numbers import catalan_numbers as py_func


@pytest.mark.parametrize("n", [1, 10, 100, 500])
def test_catalan_numbers_equivalence(n):
    assert py_func(n) == cy_func(n)
