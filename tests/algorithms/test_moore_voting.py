"""Test moore_voting equivalence."""

import pytest

from cnake_charmer.cy.algorithms.moore_voting import moore_voting as cy_func
from cnake_charmer.py.algorithms.moore_voting import moore_voting as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_moore_voting_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result
