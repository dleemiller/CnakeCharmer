"""Test count inversions equivalence."""

import pytest

from cnake_charmer.cy.algorithms.count_inversions import count_inversions as cy_count_inversions
from cnake_charmer.py.algorithms.count_inversions import count_inversions as py_count_inversions


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_count_inversions_equivalence(n):
    assert py_count_inversions(n) == cy_count_inversions(n)
