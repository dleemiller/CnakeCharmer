"""Test xor_cipher_checksum equivalence."""

import pytest

from cnake_data.cy.cryptography.xor_cipher_checksum import xor_cipher_checksum as cy_func
from cnake_data.py.cryptography.xor_cipher_checksum import xor_cipher_checksum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_xor_cipher_checksum_equivalence(n):
    assert py_func(n) == cy_func(n)
