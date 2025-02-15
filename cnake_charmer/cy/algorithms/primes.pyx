# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython implementation of prime number generator.

This module provides a fast Cython implementation for generating prime numbers
up to a specified limit. It uses static typing and array-based operations
for improved performance.

Note:
    The implementation is limited to generating up to 1000 prime numbers.
    For larger sequences, consider using a more scalable algorithm.

Example:
    >>> from cnake_charmer.cy.algorithms.primes import primes
    >>> primes(10)
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
"""

from cnake_charmer.benchmarks import cython_benchmark
import cython

# Import C array functionality
from cpython cimport array
from libc.stdlib cimport malloc, free


@cython_benchmark(args=(300,))
def primes(nb_primes: cython.int) -> list:
    """Generate a list of prime numbers.

    This function implements a basic prime sieve algorithm optimized
    for Cython execution. It uses static typing and C-style arrays
    for better performance.

    Args:
        nb_primes: Number of prime numbers to generate.
            Must be positive and will be capped at 1000.

    Returns:
        list: Python list containing the generated prime numbers.

    Raises:
        MemoryError: If unable to allocate memory for the prime array.
        
    Note:
        The function automatically caps the output at 1000 primes
        to prevent buffer overflow.
    """
    cdef:
        int i                    # Loop counter
        int[1000] p             # Array to store prime numbers
        int len_p = 0           # Current number of primes found
        int n = 2               # Number being tested for primality

    # Cap the maximum number of primes for safety
    if nb_primes > 1000:
        nb_primes = 1000

    # Initialize array in pure Python mode
    if not cython.compiled:
        p = [0] * 1000

    # Main prime finding loop
    while len_p < nb_primes:
        # Test if n is divisible by any previously found prime
        for i in p[:len_p]:
            if n % i == 0:
                break
        else:
            # n is prime - add to our array
            p[len_p] = n
            len_p += 1
        n += 1

    # Convert C array to Python list for return
    return [prime for prime in p[:len_p]]
