"""Test bloom_filter equivalence."""

import pytest

from cnake_charmer.cy.algorithms.bloom_filter import bloom_filter as cy_bloom_filter
from cnake_charmer.py.algorithms.bloom_filter import bloom_filter as py_bloom_filter


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_bloom_filter_equivalence(n):
    assert py_bloom_filter(n) == cy_bloom_filter(n)
