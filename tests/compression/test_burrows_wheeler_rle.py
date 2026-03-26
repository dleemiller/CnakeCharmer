"""Test burrows_wheeler_rle equivalence."""

import pytest

from cnake_charmer.cy.compression.burrows_wheeler_rle import burrows_wheeler_rle as cy_func
from cnake_charmer.py.compression.burrows_wheeler_rle import burrows_wheeler_rle as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_burrows_wheeler_rle_equivalence(n):
    assert py_func(n) == cy_func(n)
