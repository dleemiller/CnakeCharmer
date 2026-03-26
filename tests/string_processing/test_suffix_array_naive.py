"""Test suffix_array_naive equivalence."""

import pytest

from cnake_charmer.cy.string_processing.suffix_array_naive import suffix_array_naive as cy_func
from cnake_charmer.py.string_processing.suffix_array_naive import suffix_array_naive as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_suffix_array_naive_equivalence(n):
    assert py_func(n) == cy_func(n)
