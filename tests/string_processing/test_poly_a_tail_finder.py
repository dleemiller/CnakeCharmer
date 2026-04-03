"""Test poly_a_tail_finder equivalence."""

import pytest

from cnake_charmer.cy.string_processing.poly_a_tail_finder import poly_a_tail_finder as cy_func
from cnake_charmer.py.string_processing.poly_a_tail_finder import poly_a_tail_finder as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_poly_a_tail_finder_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
