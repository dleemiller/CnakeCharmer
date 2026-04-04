"""Test bit_count_range equivalence."""

import pytest

from cnake_data.cy.algorithms.bit_count_range import bit_count_range as cy_bit_count_range
from cnake_data.py.algorithms.bit_count_range import bit_count_range as py_bit_count_range


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_bit_count_range_equivalence(n):
    assert py_bit_count_range(n) == cy_bit_count_range(n)
