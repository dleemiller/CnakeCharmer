# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute a moving average with a fixed window using a C-contiguous typed memoryview.

Keywords: dsp, moving average, smoothing, signal processing, typed memoryview, contiguous, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def contig_moving_avg(int n):
    """Compute moving average with window=7 using C-contiguous double[::1] memoryview."""
    cdef int window = 7
    cdef int i, j, out_len
    cdef double s, total, inv_w

    cdef double *ptr = <double *>malloc(n * sizeof(double))
    if not ptr:
        raise MemoryError()

    cdef double[::1] signal = <double[:n]>ptr

    for i in range(n):
        signal[i] = ((i * 61 + 29) % 1000) / 50.0

    out_len = n - window + 1
    total = 0.0
    inv_w = 1.0 / window

    for i in range(out_len):
        s = 0.0
        for j in range(window):
            s += signal[i + j]
        total += s * inv_w

    free(ptr)
    return total
