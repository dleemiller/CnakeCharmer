# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Elementwise addition of two f32 tensors.

Keywords: elementwise, add, neural network, tensor, f32, cython
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def elementwise_add(int n):
    """Add two tensors element-wise and return sum."""
    cdef float *a = <float *>malloc(n * sizeof(float))
    cdef float *b = <float *>malloc(n * sizeof(float))
    cdef float *out = <float *>malloc(n * sizeof(float))
    if not a or not b or not out:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0

    for i in range(n):
        a[i] = ((i * 31 + 17) % 1000) * 0.01
        b[i] = ((i * 13 + 7) % 500) * 0.01

    for i in range(n):
        out[i] = a[i] + b[i]

    for i in range(n):
        total += out[i]

    free(a); free(b); free(out)
    return total
