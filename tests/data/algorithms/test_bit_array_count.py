"""Test bit_array_count equivalence."""

import pytest

from cnake_data.cy.algorithms.bit_array_count import bit_array_count as cy_func
from cnake_data.py.algorithms.bit_array_count import bit_array_count as py_func


@pytest.mark.parametrize("n", [100, 1000, 5000, 10000])
def test_bit_array_count_equivalence(n):
    assert py_func(n) == cy_func(n)
