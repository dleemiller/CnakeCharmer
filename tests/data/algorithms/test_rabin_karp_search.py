"""Test rabin_karp_search equivalence."""

import pytest

from cnake_data.cy.algorithms.rabin_karp_search import rabin_karp_search as cy_func
from cnake_data.py.algorithms.rabin_karp_search import rabin_karp_search as py_func


@pytest.mark.parametrize("n", [100, 500, 1000, 5000])
def test_rabin_karp_search_equivalence(n):
    assert py_func(n) == cy_func(n)
