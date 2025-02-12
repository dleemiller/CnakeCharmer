# cython: language_level=3
"""
FizzBuzz Implementation in Cython.

This module provides a performance-optimized FizzBuzz example as part of the living dataset.

Keywords: fizzbuzz, leetcode, cython, benchmark, example

"""

from libc.stdio cimport printf
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def fizzbuzz(int n):
    """Generate the FizzBuzz sequence for numbers from 1 to n.

    Args:
        n (int): The upper limit of numbers to process.

    Returns:
        list: A list where multiples of 3 are replaced with 'Fizz',
        multiples of 5 with 'Buzz', and multiples of both with 'FizzBuzz'.
    """
    cdef int i
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


def main():
    import sys
    cdef int n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    result = fizzbuzz(n)
    for s in result:
        print(s)

