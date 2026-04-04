"""Test rle_encode equivalence."""

import pytest

from cnake_charmer.cy.compression.rle_encode import rle_encode as cy_rle_encode
from cnake_charmer.py.compression.rle_encode import rle_encode as py_rle_encode


@pytest.mark.parametrize("n", [100, 5000, 50000, 500000])
def test_rle_encode_equivalence(n):
    py_result = py_rle_encode(n)
    cy_result = cy_rle_encode(n)
    assert py_result == cy_result, f"Mismatch at n={n}: py={py_result}, cy={cy_result}"
