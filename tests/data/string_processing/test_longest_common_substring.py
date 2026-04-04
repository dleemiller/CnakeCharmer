"""Test longest_common_substring equivalence."""

import pytest

from cnake_data.cy.string_processing.longest_common_substring import (
    longest_common_substring as cy_func,
)
from cnake_data.py.string_processing.longest_common_substring import (
    longest_common_substring as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_longest_common_substring_equivalence(n):
    assert py_func(n) == cy_func(n)
