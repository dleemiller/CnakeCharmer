"""Test huffman_encode equivalence."""

import pytest

from cnake_data.cy.compression.huffman_encode import huffman_encode as cy_func
from cnake_data.py.compression.huffman_encode import huffman_encode as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_huffman_encode_equivalence(n):
    assert py_func(n) == cy_func(n)
