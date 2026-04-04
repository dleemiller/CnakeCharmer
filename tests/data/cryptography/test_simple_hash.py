"""Test simple_hash equivalence."""

import pytest

from cnake_data.cy.cryptography.simple_hash import simple_hash as cy_func
from cnake_data.py.cryptography.simple_hash import simple_hash as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_simple_hash_equivalence(n):
    assert py_func(n) == cy_func(n)
