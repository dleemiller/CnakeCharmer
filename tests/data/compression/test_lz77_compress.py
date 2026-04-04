"""Test lz77_compress equivalence."""

import pytest

from cnake_data.cy.compression.lz77_compress import lz77_compress as cy_func
from cnake_data.py.compression.lz77_compress import lz77_compress as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_lz77_compress_equivalence(n):
    assert py_func(n) == cy_func(n)
