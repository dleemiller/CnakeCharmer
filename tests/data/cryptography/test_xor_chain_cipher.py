"""Test xor_chain_cipher equivalence."""

import pytest

from cnake_data.cy.cryptography.xor_chain_cipher import xor_chain_cipher as cy_func
from cnake_data.py.cryptography.xor_chain_cipher import xor_chain_cipher as py_func


@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_xor_chain_cipher_equivalence(n):
    assert py_func(n) == cy_func(n)
