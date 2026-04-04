"""Test xtea_encrypt equivalence."""

import pytest

from cnake_data.cy.cryptography.xtea_encrypt import xtea_encrypt as cy_func
from cnake_data.py.cryptography.xtea_encrypt import xtea_encrypt as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_xtea_encrypt_equivalence(n):
    assert py_func(n) == cy_func(n)
