"""Test aes_sbox equivalence."""

import pytest

from cnake_data.cy.cryptography.aes_sbox import aes_sbox as cy_func
from cnake_data.py.cryptography.aes_sbox import aes_sbox as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_aes_sbox_equivalence(n):
    assert py_func(n) == cy_func(n)
