# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Deterministic dropout mask on f32 tensor (basic Cython, scalar loop).

Keywords: dropout, mask, neural network, tensor, f32, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def dropout_mask(int n):
    """Apply deterministic dropout mask with p=0.1, return sum."""
    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0
    cdef float p = 0.1
    cdef float scale = 1.0 / (1.0 - p)
    cdef int threshold = <int>(p * 100)

    # Generate input
    for i in range(n):
        data[i] = sin(i * 0.01) * 10.0

    # Apply dropout mask
    for i in range(n):
        if (i * 7 + 3) % 100 >= threshold:
            total += data[i] * scale
        # else: output is 0

    free(data)
    return total
