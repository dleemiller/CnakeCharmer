"""Test salsa20_quarter equivalence."""

import pytest

from cnake_charmer.cy.cryptography.salsa20_quarter import salsa20_quarter as cy_func
from cnake_charmer.py.cryptography.salsa20_quarter import salsa20_quarter as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_salsa20_quarter_equivalence(n):
    assert py_func(n) == cy_func(n)
