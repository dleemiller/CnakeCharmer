"""Test stl_vector_sieve equivalence between Python and Cython."""

import pytest

from cnake_charmer.cy.algorithms.stl_vector_sieve import stl_vector_sieve as cy_func
from cnake_charmer.py.algorithms.stl_vector_sieve import stl_vector_sieve as py_func


@pytest.mark.parametrize("n", [1000, 50000, 500000])
def test_stl_vector_sieve_equivalence(n):
    py_r = py_func(n)
    cy_r = cy_func(n)
    assert py_r == cy_r
