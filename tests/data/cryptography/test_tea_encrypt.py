"""Test tea_encrypt equivalence."""

import pytest

from cnake_data.cy.cryptography.tea_encrypt import tea_encrypt as cy_func
from cnake_data.py.cryptography.tea_encrypt import tea_encrypt as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_tea_encrypt_equivalence(n):
    assert py_func(n) == cy_func(n)
