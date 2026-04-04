"""Test palindrome_count equivalence."""

import pytest

from cnake_data.cy.string_processing.palindrome_count import palindrome_count as cy_func
from cnake_data.py.string_processing.palindrome_count import palindrome_count as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_palindrome_count_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
