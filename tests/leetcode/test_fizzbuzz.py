"""
Test for parallel Python and Cython implementations of FizzBuzz.

Keywords: fizzbuzz, leetcode, python, cython, test, benchmark
"""

import pytest

# Import the Python FizzBuzz implementation.
from cnake_charmer.py.leetcode.fizzbuzz import fizzbuzz as py_fizzbuzz

# Import the Cython FizzBuzz implementation.
from cnake_charmer.cy.leetcode.fizzbuzz import fizzbuzz as cy_fizzbuzz


@pytest.mark.parametrize("n", [10, 15, 20, 30])
def test_fizzbuzz_equivalence(n):
    """
    Compare the output of the Python and Cython implementations for a given n.
    """
    py_result = py_fizzbuzz(n)
    cy_result = cy_fizzbuzz(n)
    assert py_result == cy_result, f"Results differ for n={n}"
