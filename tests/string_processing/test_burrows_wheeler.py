"""Test burrows_wheeler equivalence."""

import pytest

from cnake_charmer.cy.string_processing.burrows_wheeler import burrows_wheeler as cy_func
from cnake_charmer.py.string_processing.burrows_wheeler import burrows_wheeler as py_func


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_burrows_wheeler_equivalence(n):
    py_result = py_func(n)
    cy_result = cy_func(n)
    assert py_result == cy_result, f"Mismatch at n={n}: {py_result} vs {cy_result}"
