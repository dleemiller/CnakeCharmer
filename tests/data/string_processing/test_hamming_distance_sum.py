"""Test hamming_distance_sum equivalence."""

import pytest

from cnake_data.cy.string_processing.hamming_distance_sum import hamming_distance_sum as cy_func
from cnake_data.py.string_processing.hamming_distance_sum import hamming_distance_sum as py_func


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_hamming_distance_sum_equivalence(n):
    assert py_func(n) == cy_func(n)
