"""Test longest_zigzag equivalence."""

import pytest

from cnake_data.cy.dynamic_programming.longest_zigzag import longest_zigzag as cy_func
from cnake_data.py.dynamic_programming.longest_zigzag import longest_zigzag as py_func


@pytest.mark.parametrize("n", [1, 10, 100, 1000, 10000])
def test_longest_zigzag_equivalence(n):
    assert py_func(n) == cy_func(n)
