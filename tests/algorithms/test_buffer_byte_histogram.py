"""Test buffer_byte_histogram equivalence."""

import pytest

from cnake_charmer.cy.algorithms.buffer_byte_histogram import (
    buffer_byte_histogram as cy_buffer_byte_histogram,
)
from cnake_charmer.py.algorithms.buffer_byte_histogram import (
    buffer_byte_histogram as py_buffer_byte_histogram,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 100000])
def test_buffer_byte_histogram_equivalence(n):
    assert py_buffer_byte_histogram(n) == cy_buffer_byte_histogram(n)
