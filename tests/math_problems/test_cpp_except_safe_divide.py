"""Test cpp_except_safe_divide equivalence between Python and Cython."""

import pytest

from cnake_charmer.cy.math_problems.cpp_except_safe_divide import cpp_except_safe_divide as cy_func
from cnake_charmer.py.math_problems.cpp_except_safe_divide import cpp_except_safe_divide as py_func


@pytest.mark.parametrize("n", [1000, 50000, 500000])
def test_cpp_except_safe_divide_equivalence(n):
    py_r = py_func(n)
    cy_r = cy_func(n)
    assert py_r == cy_r
