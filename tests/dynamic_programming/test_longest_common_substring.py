"""Test longest_common_substring equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.longest_common_substring import (
    longest_common_substring as cy_func,
)
from cnake_charmer.py.dynamic_programming.longest_common_substring import (
    longest_common_substring as py_func,
)


@pytest.mark.parametrize("n", [10, 50, 100, 500])
def test_longest_common_substring_equivalence(n):
    assert py_func(n) == cy_func(n)
