# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sigmoid activation function.

Keywords: sigmoid, activation, neural network, elementwise, exp, cython
"""

from libc.math cimport exp
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def sigmoid(int n):
    """Apply sigmoid to n values and return sum."""
    cdef double total = 0.0
    cdef int i
    cdef double v

    for i in range(n):
        v = ((i * 17 + 5) % 1000) / 100.0 - 5.0
        total += 1.0 / (1.0 + exp(-v))
    return total
