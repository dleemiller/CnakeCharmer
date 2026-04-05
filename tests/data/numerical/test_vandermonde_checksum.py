"""Test vandermonde_checksum equivalence."""

import pytest

from cnake_data.cy.numerical.vandermonde_checksum import vandermonde_checksum as cy_func
from cnake_data.py.numerical.vandermonde_checksum import vandermonde_checksum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_vandermonde_checksum_equivalence(n):
    assert py_func(n) == cy_func(n)
