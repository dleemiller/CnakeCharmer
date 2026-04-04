"""Test forward_decl_tree_sum equivalence."""

import pytest

from cnake_data.cy.algorithms.forward_decl_tree_sum import forward_decl_tree_sum as cy_func
from cnake_data.py.algorithms.forward_decl_tree_sum import forward_decl_tree_sum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_forward_decl_tree_sum_equivalence(n):
    assert py_func(n) == cy_func(n)
