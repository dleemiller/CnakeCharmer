# cython: boundscheck=False, wraparound=False, language_level=3
"""
FizzBuzz Implementation in Cython.

This module provides a performance-optimized FizzBuzz example as part of the living dataset.

Keywords: fizzbuzz, leetcode, cython, benchmark, example

"""
from cnake_charmer.benchmarks import cython_benchmark
import cython


@cython_benchmark(syntax="pp", args=(10000,))
def fizzbuzz(n: cython.int) -> list[cython.str]:
    # Preallocate the list to avoid repeated appends.
    results: list[cython.str] = [None] * n
    i: cython.int

    for i in range(1, n + 1):
        if i % 15 == 0:
            results[i - 1] = "FizzBuzz"
        elif i % 3 == 0:
            results[i - 1] = "Fizz"
        elif i % 5 == 0:
            results[i - 1] = "Buzz"
        else:
            # Convert the integer to a Python string.
            results[i - 1] = str(i)

    return results
