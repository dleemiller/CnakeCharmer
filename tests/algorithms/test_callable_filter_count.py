"""Test callable_filter_count equivalence."""

import pytest

from cnake_charmer.cy.algorithms.callable_filter_count import callable_filter_count as cy_func
from cnake_charmer.py.algorithms.callable_filter_count import callable_filter_count as py_func


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_callable_filter_count_equivalence(n):
    assert py_func(n) == cy_func(n)
