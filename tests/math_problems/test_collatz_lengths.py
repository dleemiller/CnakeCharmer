"""Test Collatz lengths equivalence."""

import pytest

from cnake_charmer.cy.math_problems.collatz_lengths import collatz_lengths as cy_collatz_lengths
from cnake_charmer.py.math_problems.collatz_lengths import collatz_lengths as py_collatz_lengths


@pytest.mark.parametrize("n", [10, 100, 1000, 5000])
def test_collatz_lengths_equivalence(n):
    assert py_collatz_lengths(n) == cy_collatz_lengths(n)
