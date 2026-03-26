"""Test chacha20_block equivalence."""

import pytest

from cnake_charmer.cy.cryptography.chacha20_block import chacha20_block as cy_func
from cnake_charmer.py.cryptography.chacha20_block import chacha20_block as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_chacha20_block_equivalence(n):
    assert py_func(n) == cy_func(n)
