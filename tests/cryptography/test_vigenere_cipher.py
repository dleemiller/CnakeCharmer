"""Test vigenere_cipher equivalence."""

import pytest

from cnake_charmer.cy.cryptography.vigenere_cipher import vigenere_cipher as cy_func
from cnake_charmer.py.cryptography.vigenere_cipher import vigenere_cipher as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_vigenere_cipher_equivalence(n):
    assert py_func(n) == cy_func(n)
