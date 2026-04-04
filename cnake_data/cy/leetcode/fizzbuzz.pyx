# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
FizzBuzz Implementation in Cython.

This module provides a performance-optimized FizzBuzz example as part of the living dataset.

Keywords: fizzbuzz, leetcode, cython, benchmark, example

"""
from cnake_data.benchmarks import cython_benchmark

from cpython.list cimport PyList_SET_ITEM
from cpython.ref cimport Py_INCREF
from cpython.unicode cimport PyUnicode_New, PyUnicode_1BYTE_DATA
from libc.stdint cimport uint8_t


cdef inline object _uint_to_str(unsigned int v):
    """Convert a non-negative integer to a Python str via direct ASCII write."""
    cdef unsigned int tmp = v
    cdef int ndigits, i
    cdef object s
    cdef uint8_t *data

    # Count digits
    if tmp < 10:
        ndigits = 1
    elif tmp < 100:
        ndigits = 2
    elif tmp < 1000:
        ndigits = 3
    elif tmp < 10000:
        ndigits = 4
    elif tmp < 100000:
        ndigits = 5
    elif tmp < 1000000:
        ndigits = 6
    elif tmp < 10000000:
        ndigits = 7
    else:
        ndigits = 8

    # PyUnicode_New with maxchar=127 creates a compact ASCII (1-byte kind) object
    s = PyUnicode_New(ndigits, 127)
    data = <uint8_t *>PyUnicode_1BYTE_DATA(s)

    # Write digits right-to-left
    i = ndigits - 1
    while tmp >= 10:
        data[i] = 48 + (tmp % 10)
        tmp = tmp // 10
        i -= 1
    data[i] = 48 + tmp
    return s


@cython_benchmark(syntax="cy", args=(1000000,))
def fizzbuzz(int n):
    """FizzBuzz written in Cython syntax."""
    cdef list result = [None] * n
    cdef int i
    cdef object s

    cdef object fizz = "Fizz"
    cdef object buzz = "Buzz"
    cdef object fizzbuzz_str = "FizzBuzz"

    for i in range(1, n + 1):
        if i % 15 == 0:
            s = fizzbuzz_str
        elif i % 3 == 0:
            s = fizz
        elif i % 5 == 0:
            s = buzz
        else:
            s = _uint_to_str(i)
        Py_INCREF(s)
        PyList_SET_ITEM(result, i - 1, s)

    return result
