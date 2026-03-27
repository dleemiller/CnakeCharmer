"""Test interval_overlap_count equivalence."""

import pytest

from cnake_charmer.cy.algorithms.interval_overlap_count import interval_overlap_count as cy_func
from cnake_charmer.py.algorithms.interval_overlap_count import interval_overlap_count as py_func


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_interval_overlap_count_equivalence(n):
    assert py_func(n) == cy_func(n)
