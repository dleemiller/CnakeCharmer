"""Test caesar_cipher equivalence."""

import pytest

from cnake_data.cy.cryptography.caesar_cipher import caesar_cipher as cy_func
from cnake_data.py.cryptography.caesar_cipher import caesar_cipher as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_caesar_cipher_equivalence(n):
    assert py_func(n) == cy_func(n)
