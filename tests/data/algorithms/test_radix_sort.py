"""Test radix sort equivalence."""

import pytest

from cnake_data.cy.algorithms.radix_sort import radix_sort as cy_radix_sort
from cnake_data.py.algorithms.radix_sort import radix_sort as py_radix_sort


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_radix_sort_equivalence(n):
    py_result = py_radix_sort(n)
    cy_result = cy_radix_sort(n)
    assert py_result == cy_result, f"Mismatch at n={n}"
