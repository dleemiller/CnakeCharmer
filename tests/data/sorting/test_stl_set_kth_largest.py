"""Test stl_set_kth_largest equivalence between Python and Cython."""

import pytest

from cnake_data.cy.sorting.stl_set_kth_largest import stl_set_kth_largest as cy_func
from cnake_data.py.sorting.stl_set_kth_largest import stl_set_kth_largest as py_func


@pytest.mark.parametrize("n", [2000, 20000, 100000])
def test_stl_set_kth_largest_equivalence(n):
    py_r = py_func(n)
    cy_r = cy_func(n)
    assert py_r == cy_r
