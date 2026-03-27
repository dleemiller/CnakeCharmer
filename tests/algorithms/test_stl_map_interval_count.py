"""Test stl_map_interval_count equivalence between Python and Cython."""

import pytest

from cnake_charmer.cy.algorithms.stl_map_interval_count import stl_map_interval_count as cy_func
from cnake_charmer.py.algorithms.stl_map_interval_count import stl_map_interval_count as py_func


@pytest.mark.parametrize("n", [100, 10000, 200000])
def test_stl_map_interval_count_equivalence(n):
    py_r = py_func(n)
    cy_r = cy_func(n)
    assert py_r == cy_r
