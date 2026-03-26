"""Test suffix_array_lcp equivalence."""

import pytest

from cnake_charmer.cy.string_processing.suffix_array_lcp import suffix_array_lcp as cy_func
from cnake_charmer.py.string_processing.suffix_array_lcp import suffix_array_lcp as py_func


@pytest.mark.parametrize("n", [1, 10, 100, 500, 1000])
def test_suffix_array_lcp_equivalence(n):
    assert py_func(n) == cy_func(n)
