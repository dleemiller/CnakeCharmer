# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Batch normalization.

Keywords: batch norm, normalization, neural network, statistics, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def batch_norm(int n):
    """Batch normalize n values and return sum of normalized values."""
    cdef double *vals = <double *>malloc(n * sizeof(double))
    if not vals:
        raise MemoryError()

    cdef int i
    cdef double mean = 0.0
    cdef double var = 0.0
    cdef double diff, inv_std, total, v

    # Fill values and compute mean
    for i in range(n):
        v = ((i * 17 + 5) % 1000) / 10.0
        vals[i] = v
        mean += v
    mean /= n

    # Compute variance
    for i in range(n):
        diff = vals[i] - mean
        var += diff * diff
    var /= n

    # Normalize and sum
    inv_std = 1.0 / sqrt(var + 1e-5)
    total = 0.0
    for i in range(n):
        total += (vals[i] - mean) * inv_std

    free(vals)
    return total
