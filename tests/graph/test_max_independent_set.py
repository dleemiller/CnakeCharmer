"""Test max_independent_set equivalence."""

import pytest

from cnake_charmer.cy.graph.max_independent_set import max_independent_set as cy_func
from cnake_charmer.py.graph.max_independent_set import max_independent_set as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_max_independent_set_equivalence(n):
    assert py_func(n) == cy_func(n)
