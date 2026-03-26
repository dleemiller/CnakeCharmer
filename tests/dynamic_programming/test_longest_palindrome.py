"""Test longest_palindrome equivalence."""

import pytest

from cnake_charmer.cy.dynamic_programming.longest_palindrome import (
    longest_palindrome as cy_longest_palindrome,
)
from cnake_charmer.py.dynamic_programming.longest_palindrome import (
    longest_palindrome as py_longest_palindrome,
)


@pytest.mark.parametrize("n", [10, 50, 200, 500])
def test_longest_palindrome_equivalence(n):
    assert py_longest_palindrome(n) == cy_longest_palindrome(n)
