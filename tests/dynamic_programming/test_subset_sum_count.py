"""Test subset_sum_count equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.subset_sum_count import subset_sum_count as cy_func
from cnake_charmer.py.dynamic_programming.subset_sum_count import subset_sum_count as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 200])
def test_subset_sum_count_equivalence(n):
    assert py_func(n) == cy_func(n)
