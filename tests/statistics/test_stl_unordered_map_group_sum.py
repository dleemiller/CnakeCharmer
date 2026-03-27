"""Test stl_unordered_map_group_sum equivalence between Python and Cython."""

import pytest

from cnake_charmer.cy.statistics.stl_unordered_map_group_sum import (
    stl_unordered_map_group_sum as cy_func,
)
from cnake_charmer.py.statistics.stl_unordered_map_group_sum import (
    stl_unordered_map_group_sum as py_func,
)


@pytest.mark.parametrize("n", [1000, 100000, 1000000])
def test_stl_unordered_map_group_sum_equivalence(n):
    py_r = py_func(n)
    cy_r = cy_func(n)
    assert py_r == cy_r
