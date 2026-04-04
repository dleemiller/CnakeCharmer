"""Test longest increasing subsequence equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.longest_increasing_subsequence import (
    longest_increasing_subsequence as cy_lis,
)
from cnake_data.py.dynamic_programming.longest_increasing_subsequence import (
    longest_increasing_subsequence as py_lis,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_longest_increasing_subsequence_equivalence(n):
    assert py_lis(n) == cy_lis(n)
