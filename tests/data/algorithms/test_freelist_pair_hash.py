"""Test freelist_pair_hash equivalence."""

import pytest

from cnake_data.cy.algorithms.freelist_pair_hash import freelist_pair_hash as cy_func
from cnake_data.py.algorithms.freelist_pair_hash import freelist_pair_hash as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_freelist_pair_hash_equivalence(n):
    assert py_func(n) == cy_func(n)
