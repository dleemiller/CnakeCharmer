"""Test lz77_match_chain equivalence."""

import pytest

from cnake_charmer.cy.compression.lz77_match_chain import lz77_match_chain as cy_func
from cnake_charmer.py.compression.lz77_match_chain import lz77_match_chain as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_lz77_match_chain_equivalence(n):
    assert py_func(n) == cy_func(n)
