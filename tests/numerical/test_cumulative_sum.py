"""Test cumulative sum equivalence."""

import pytest

from cnake_charmer.cy.numerical.cumulative_sum import cumulative_sum as cy_cumulative_sum
from cnake_charmer.py.numerical.cumulative_sum import cumulative_sum as py_cumulative_sum


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_cumulative_sum_equivalence(n):
    assert py_cumulative_sum(n) == cy_cumulative_sum(n)
