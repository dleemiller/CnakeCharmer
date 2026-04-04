"""Test count substrings equivalence."""

import pytest

from cnake_data.cy.string_processing.count_substrings import (
    count_substrings as cy_count_substrings,
)
from cnake_data.py.string_processing.count_substrings import (
    count_substrings as py_count_substrings,
)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_count_substrings_equivalence(n):
    assert py_count_substrings(n) == cy_count_substrings(n)
