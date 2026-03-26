"""Test wyhash equivalence."""

import pytest

from cnake_charmer.cy.cryptography.wyhash import wyhash as cy_func
from cnake_charmer.py.cryptography.wyhash import wyhash as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_wyhash_equivalence(n):
    assert py_func(n) == cy_func(n)
