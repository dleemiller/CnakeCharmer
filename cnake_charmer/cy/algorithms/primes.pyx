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


@cython_benchmark(args=(300,))
def primes(nb_primes: cython.int):
    """Generate a list of prime numbers.

    Args:
        nb_primes: Number of prime numbers to generate.
            Must be positive and will be capped at 1000.

    Returns:
        list: Python list containing the generated prime numbers.

    Note:
        The function automatically caps the output at 1000 primes
        to prevent buffer overflow.
    """
    i: cython.int
    p: cython.int[1000]

    if nb_primes > 1000:
        nb_primes = 1000

    if not cython.compiled:  # Only if regular Python is running
        p = [0] * 1000       # Make p work almost like a C array

    len_p: cython.int = 0  # The current number of elements in p.
    n: cython.int = 2
    while len_p < nb_primes:
        # Is n prime?
        for i in p[:len_p]:
            if n % i == 0:
                break

        # If no break occurred in the loop, we have a prime.
        else:
            p[len_p] = n
            len_p += 1
        n += 1

    return [prime for prime in p[:len_p]]
