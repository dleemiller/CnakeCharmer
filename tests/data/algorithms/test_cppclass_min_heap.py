"""Test cppclass_min_heap equivalence between Python and Cython."""

import pytest

from cnake_data.cy.algorithms.cppclass_min_heap import cppclass_min_heap as cy_func
from cnake_data.py.algorithms.cppclass_min_heap import cppclass_min_heap as py_func


@pytest.mark.parametrize("n", [1000, 50000, 500000])
def test_cppclass_min_heap_equivalence(n):
    py_r = py_func(n)
    cy_r = cy_func(n)
    assert py_r == cy_r
