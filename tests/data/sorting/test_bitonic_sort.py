"""Test bitonic_sort equivalence."""

import pytest

from cnake_data.cy.sorting.bitonic_sort import bitonic_sort as cy_func
from cnake_data.py.sorting.bitonic_sort import bitonic_sort as py_func


@pytest.mark.parametrize("n", [8, 64, 256, 1024])
def test_bitonic_sort_equivalence(n):
    assert py_func(n) == cy_func(n)
