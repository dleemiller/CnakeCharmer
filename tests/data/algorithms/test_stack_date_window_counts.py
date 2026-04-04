"""Test stack_date_window_counts equivalence."""

import pytest

from cnake_data.cy.algorithms.stack_date_window_counts import stack_date_window_counts as cy_func
from cnake_data.py.algorithms.stack_date_window_counts import stack_date_window_counts as py_func


@pytest.mark.parametrize("args", [(2000, 3, 0), (1995, 6, 1), (2001, 2, 2), (1993, 10, 2)])
def test_stack_date_window_counts_equivalence(args):
    assert py_func(*args) == cy_func(*args)
