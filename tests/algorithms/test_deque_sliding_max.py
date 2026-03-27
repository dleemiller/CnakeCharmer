"""Test deque_sliding_max equivalence."""

import pytest

from cnake_charmer.cy.algorithms.deque_sliding_max import deque_sliding_max as cy_func
from cnake_charmer.py.algorithms.deque_sliding_max import deque_sliding_max as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_deque_sliding_max_equivalence(n):
    assert py_func(n) == cy_func(n)
