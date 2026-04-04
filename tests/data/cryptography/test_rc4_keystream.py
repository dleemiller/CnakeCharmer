"""Test rc4_keystream equivalence."""

import pytest

from cnake_data.cy.cryptography.rc4_keystream import rc4_keystream as cy_func
from cnake_data.py.cryptography.rc4_keystream import rc4_keystream as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_rc4_keystream_equivalence(n):
    assert py_func(n) == cy_func(n)
