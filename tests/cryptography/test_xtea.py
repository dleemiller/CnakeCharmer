"""Test xtea equivalence."""

import pytest

from cnake_charmer.cy.cryptography.xtea import xtea as cy_func
from cnake_charmer.py.cryptography.xtea import xtea as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_xtea_equivalence(n):
    assert py_func(n) == cy_func(n)
