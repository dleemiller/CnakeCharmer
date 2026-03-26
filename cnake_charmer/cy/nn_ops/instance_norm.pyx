# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Instance normalization on f32 tensor (basic Cython, scalar loop).

Keywords: instance_norm, normalization, neural network, tensor, f32, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, sqrt
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def instance_norm(int n):
    """Instance normalization per-channel, return sum of output."""
    cdef int channels = 16
    cdef int spatial = n // channels
    cdef float eps = 1e-5

    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int c, s, offset
    cdef double mean, var, diff, inv_std
    cdef double total = 0.0

    # Generate input
    for c in range(n):
        data[c] = sin(c * 0.01) * 10.0

    for c in range(channels):
        offset = c * spatial

        # Mean
        mean = 0.0
        for s in range(spatial):
            mean += data[offset + s]
        mean /= spatial

        # Variance
        var = 0.0
        for s in range(spatial):
            diff = data[offset + s] - mean
            var += diff * diff
        var /= spatial

        # Normalize
        inv_std = 1.0 / sqrt(var + eps)
        for s in range(spatial):
            total += (data[offset + s] - mean) * inv_std

    free(data)
    return total
