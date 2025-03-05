# cython: boundscheck=False, wraparound=False, language_level=3
"""
FizzBuzz Implementation in Cython.

This module provides a performance-optimized FizzBuzz example as part of the living dataset.

Keywords: fizzbuzz, leetcode, cython, benchmark, example

"""
import cython

def fizzbuzz(int n):
    """FizzBuzz written in Cython syntax."""
    cdef list result = [None] * n
    cdef int i

    for i in range(1, n+1):
        if i % 15 == 0:
            result[i - 1] = "FizzBuzz"
        elif i % 3 == 0:
            result[i - 1] = "Fizz"
        elif i % 5 == 0:
            result[i - 1] = "Buzz"
        else:
            result[i - 1] = str(i)

    return result

