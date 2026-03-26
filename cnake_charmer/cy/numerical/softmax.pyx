# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute grouped softmax and sum all outputs (Cython-optimized).

Keywords: numerical, softmax, exponential, normalization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport exp
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def softmax(int n):
    """Compute grouped softmax and sum outputs using C arrays and libc exp."""
    cdef int GROUP = 100
    cdef int num_groups = n // GROUP

    cdef double *values = <double *>malloc(n * sizeof(double))
    if not values:
        raise MemoryError()

    cdef int i, g, start
    cdef double max_val, exp_sum, total, val

    for i in range(n):
        values[i] = (i * 17 + 5) % 100 / 50.0 - 1.0

    total = 0.0
    for g in range(num_groups):
        start = g * GROUP

        # Find max for numerical stability
        max_val = values[start]
        for i in range(start + 1, start + GROUP):
            if values[i] > max_val:
                max_val = values[i]

        # Compute exp sum
        exp_sum = 0.0
        for i in range(start, start + GROUP):
            exp_sum += exp(values[i] - max_val)

        # Compute softmax and accumulate
        for i in range(start, start + GROUP):
            total += exp(values[i] - max_val) / exp_sum

    free(values)
    return total
