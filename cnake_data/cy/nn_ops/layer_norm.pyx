# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Layer normalization.

Keywords: layer norm, normalization, neural network, transformer, cython
"""

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def layer_norm(int n):
    """Layer normalize groups of 64 values and return sum."""
    cdef int group_size = 64
    cdef int num_groups = n // group_size
    cdef double epsilon = 1e-5
    cdef double total = 0.0
    cdef int g, i, offset
    cdef double mean, var, diff, inv_std, v

    cdef double *vals = <double *>malloc(group_size * sizeof(double))
    if not vals:
        raise MemoryError()

    for g in range(num_groups):
        offset = g * group_size
        # Compute values and mean
        mean = 0.0
        for i in range(group_size):
            v = (((offset + i) * 17 + 5) % 1000) / 10.0
            vals[i] = v
            mean += v
        mean /= group_size
        # Compute variance
        var = 0.0
        for i in range(group_size):
            diff = vals[i] - mean
            var += diff * diff
        var /= group_size
        inv_std = 1.0 / sqrt(var + epsilon)
        # Normalize and accumulate
        for i in range(group_size):
            total += (vals[i] - mean) * inv_std

    free(vals)
    return total
