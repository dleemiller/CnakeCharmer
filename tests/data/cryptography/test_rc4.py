"""Test rc4 equivalence."""

import pytest

from cnake_data.cy.cryptography.rc4 import rc4 as cy_func
from cnake_data.py.cryptography.rc4 import rc4 as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_rc4_equivalence(n):
    assert py_func(n) == cy_func(n)
