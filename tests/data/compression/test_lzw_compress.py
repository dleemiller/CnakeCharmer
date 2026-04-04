"""Test lzw_compress equivalence."""

import pytest

from cnake_data.cy.compression.lzw_compress import lzw_compress as cy_func
from cnake_data.py.compression.lzw_compress import lzw_compress as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_lzw_compress_equivalence(n):
    assert py_func(n) == cy_func(n)
