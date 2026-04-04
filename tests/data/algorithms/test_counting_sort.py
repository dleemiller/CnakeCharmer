"""Test counting_sort equivalence."""

import pytest

from cnake_data.cy.algorithms.counting_sort import counting_sort as cy_counting_sort
from cnake_data.py.algorithms.counting_sort import counting_sort as py_counting_sort


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_counting_sort_equivalence(n):
    py_result = py_counting_sort(n)
    cy_result = cy_counting_sort(n)
    assert py_result == cy_result, f"Mismatch at n={n}"
