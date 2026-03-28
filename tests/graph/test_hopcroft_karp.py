"""Test hopcroft_karp equivalence."""

import pytest

from cnake_charmer.cy.graph.hopcroft_karp import hopcroft_karp as cy_func
from cnake_charmer.py.graph.hopcroft_karp import hopcroft_karp as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_hopcroft_karp_equivalence(n):
    assert py_func(n) == cy_func(n)
