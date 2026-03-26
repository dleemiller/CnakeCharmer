"""Test chinese_remainder equivalence."""

import pytest

from cnake_charmer.cy.math_problems.chinese_remainder import chinese_remainder as cy_func
from cnake_charmer.py.math_problems.chinese_remainder import chinese_remainder as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_chinese_remainder_equivalence(n):
    assert py_func(n) == cy_func(n)
