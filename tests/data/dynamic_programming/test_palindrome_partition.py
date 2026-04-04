"""Test palindrome_partition equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.palindrome_partition import (
    palindrome_partition as cy_func,
)
from cnake_data.py.dynamic_programming.palindrome_partition import (
    palindrome_partition as py_func,
)


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_palindrome_partition_equivalence(n):
    assert py_func(n) == cy_func(n)
