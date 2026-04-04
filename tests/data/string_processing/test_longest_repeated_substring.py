"""Test longest_repeated_substring equivalence."""

import pytest

from cnake_data.cy.string_processing.longest_repeated_substring import (
    longest_repeated_substring as cy_func,
)
from cnake_data.py.string_processing.longest_repeated_substring import (
    longest_repeated_substring as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_longest_repeated_substring_equivalence(n):
    assert py_func(n) == cy_func(n)
