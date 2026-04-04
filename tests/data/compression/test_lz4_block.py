"""Test lz4_block equivalence."""

import pytest

from cnake_data.cy.compression.lz4_block import lz4_block as cy_func
from cnake_data.py.compression.lz4_block import lz4_block as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_lz4_block_equivalence(n):
    assert py_func(n) == cy_func(n)
