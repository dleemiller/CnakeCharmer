"""Test huffman_frequency equivalence."""

import pytest

from cnake_data.cy.compression.huffman_frequency import huffman_frequency as cy_func
from cnake_data.py.compression.huffman_frequency import huffman_frequency as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_huffman_frequency_equivalence(n):
    assert py_func(n) == cy_func(n)
