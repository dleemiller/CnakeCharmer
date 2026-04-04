"""Test siphash equivalence."""

import pytest

from cnake_data.cy.cryptography.siphash import siphash as cy_func
from cnake_data.py.cryptography.siphash import siphash as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_siphash_equivalence(n):
    assert py_func(n) == cy_func(n)
