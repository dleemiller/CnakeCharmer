"""Test stack_naive_tree_distance equivalence."""

import pytest

from cnake_data.cy.statistics.stack_naive_tree_distance import stack_naive_tree_distance as cy_func
from cnake_data.py.statistics.stack_naive_tree_distance import stack_naive_tree_distance as py_func


@pytest.mark.parametrize("args", [(80, 3), (120, 4), (200, 4), (320, 5)])
def test_stack_naive_tree_distance_equivalence(args):
    assert py_func(*args) == cy_func(*args)
