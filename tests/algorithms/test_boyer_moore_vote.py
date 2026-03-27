"""Test boyer_moore_vote equivalence."""

import pytest

from cnake_charmer.cy.algorithms.boyer_moore_vote import boyer_moore_vote as cy_func
from cnake_charmer.py.algorithms.boyer_moore_vote import boyer_moore_vote as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_boyer_moore_vote_equivalence(n):
    assert py_func(n) == cy_func(n)
