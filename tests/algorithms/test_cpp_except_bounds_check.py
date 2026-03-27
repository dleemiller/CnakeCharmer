"""Test cpp_except_bounds_check equivalence between Python and Cython."""

import pytest

from cnake_charmer.cy.algorithms.cpp_except_bounds_check import cpp_except_bounds_check as cy_func
from cnake_charmer.py.algorithms.cpp_except_bounds_check import cpp_except_bounds_check as py_func


@pytest.mark.parametrize("n", [1000, 20000, 200000])
def test_cpp_except_bounds_check_equivalence(n):
    py_r = py_func(n)
    cy_r = cy_func(n)
    assert py_r == cy_r
