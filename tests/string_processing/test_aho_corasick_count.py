"""Test aho_corasick_count equivalence."""

import pytest

from cnake_charmer.cy.string_processing.aho_corasick_count import aho_corasick_count as cy_func
from cnake_charmer.py.string_processing.aho_corasick_count import aho_corasick_count as py_func


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_aho_corasick_count_equivalence(n):
    assert py_func(n) == cy_func(n)
