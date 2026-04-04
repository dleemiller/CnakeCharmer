# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Global average pooling on f32 tensor (basic Cython, scalar loop).

channels=64, spatial=n/64. Return sum of channel means.

Keywords: global_avg_pool, pooling, neural network, tensor, f32, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def global_avg_pool(int n):
    """Global average pool, return sum of channel means."""
    cdef int channels = 64
    cdef int spatial = n // channels

    cdef float *data = <float *>malloc(n * sizeof(float))
    if not data:
        raise MemoryError()

    cdef int c, s, offset
    cdef double channel_sum
    cdef double total = 0.0

    # Generate input
    for c in range(n):
        data[c] = sin(c * 0.01) * 10.0

    # Global average pool: mean per channel
    for c in range(channels):
        offset = c * spatial
        channel_sum = 0.0
        for s in range(spatial):
            channel_sum += data[offset + s]
        total += channel_sum / spatial

    free(data)
    return total
