"""Test cpp_sort_by_popcount equivalence between Python and Cython."""

import pytest

from cnake_data.cy.sorting.cpp_sort_by_popcount import cpp_sort_by_popcount as cy_func
from cnake_data.py.sorting.cpp_sort_by_popcount import cpp_sort_by_popcount as py_func


@pytest.mark.parametrize("n", [1000, 30000, 300000])
def test_cpp_sort_by_popcount_equivalence(n):
    py_r = py_func(n)
    cy_r = cy_func(n)
    assert py_r == cy_r
