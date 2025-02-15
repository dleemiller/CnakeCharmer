"""
Test for parallel Python and Cython implementations of FizzBuzz.

Keywords: fibonacci, leetcode, python, cython, test, benchmark
"""

import pytest

# Import the Python implementation.
from cnake_charmer.py.algorithms.fibonacci import fib as py_fibonacci

# Import the Cython implementation.
from cnake_charmer.cy.algorithms.fibonacci import fib as cy_fibonacci


@pytest.mark.parametrize("n", [1e6, 1e9, 1e12, 1e18])
def test_fibonacci_equivalence(n):
    """
    Compare the output of the Python and Cython implementations for a given n.
    """
    py_result = py_fibonacci(n)
    cy_result = cy_fibonacci(n)
    assert py_result == cy_result, f"Results differ for n={n}"
