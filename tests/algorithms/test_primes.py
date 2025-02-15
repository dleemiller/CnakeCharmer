"""Test module for comparing Python and Cython prime number implementations.

This module provides test functions to verify that the Python and Cython
implementations of prime number generation produce identical results.

Example:
    Run tests using pytest:
    $ pytest test_primes.py

Attributes:
    None

Todo:
    * Add performance benchmarking tests
    * Add edge case tests (e.g., n=0, n=1)
    * Add test cases for larger numbers

Keywords:
    primes, python, cython, test, benchmark, algorithms
"""

import pytest

# Import the Python prime number implementation
from cnake_charmer.py.algorithms.primes import primes as py_primes

# Import the Cython prime number implementation
from cnake_charmer.cy.algorithms.primes import primes as cy_primes


@pytest.mark.parametrize("n", [10, 15, 20, 30])
def test_primes_equivalence(n: int):
    """Compare outputs of Python and Cython prime number implementations.

    This test function verifies that both implementations produce the same
    results for various input values, ensuring functional equivalence.

    Args:
        n (int): The upper limit for finding prime numbers.

    Raises:
        AssertionError: If the results from Python and Cython implementations differ.

    Example:
        >>> test_primes_equivalence(10)
        # No output if test passes
        # AssertionError if implementations give different results
    """
    py_result = py_primes(n)
    cy_result = cy_primes(n)
    assert py_result == cy_result, f"Results differ for n={n}"
