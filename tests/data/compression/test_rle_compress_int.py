"""Test rle_compress_int equivalence."""

import pytest

from cnake_data.cy.compression.rle_compress_int import rle_compress_int as cy_func
from cnake_data.py.compression.rle_compress_int import rle_compress_int as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_rle_compress_int_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch: py={py_result}, cy={cy_result}"
