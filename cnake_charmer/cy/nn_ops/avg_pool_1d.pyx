# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Average pooling 1D on f32 tensor (basic Cython, scalar loop).

Average pooling with kernel=4, stride=4.

Keywords: avg_pool, pooling, neural network, tensor, f32, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def avg_pool_1d(int n):
    """Allocate f32 signal, apply avg pool kernel=4 stride=4, return sum."""
    cdef float *signal = <float *>malloc(n * sizeof(float))
    if not signal:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0
    cdef int out_len = n // 4
    cdef int base
    cdef float avg

    # Generate signal
    for i in range(n):
        signal[i] = (i * 31 + 17) % 1000 / 10.0

    # Average pool kernel=4, stride=4
    for i in range(out_len):
        base = i * 4
        avg = (signal[base] + signal[base + 1] + signal[base + 2] + signal[base + 3]) * 0.25
        total += avg

    free(signal)
    return total
