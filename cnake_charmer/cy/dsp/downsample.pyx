# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Downsample a signal by factor 4 with anti-alias FIR filter (Cython-optimized).

Applies a 21-tap low-pass FIR filter before decimation and returns the
sum of the downsampled signal.

Keywords: dsp, downsample, decimate, anti-alias, FIR, filter, cython, benchmark
"""

from libc.math cimport sin, cos, M_PI
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def downsample(int n):
    """Downsample signal by factor 4 with anti-alias filter, return sum."""
    cdef int taps = 21
    cdef int mid = 10
    cdef int factor = 4
    cdef int i, k
    cdef double acc, total, diff, cutoff
    cdef double *h = <double *>malloc(taps * sizeof(double))
    cdef double *s = <double *>malloc(n * sizeof(double))
    if not h or not s:
        raise MemoryError()

    # Design anti-alias filter
    cutoff = 1.0 / factor
    for k in range(taps):
        if k == mid:
            h[k] = cutoff
        else:
            diff = k - mid
            h[k] = sin(M_PI * cutoff * diff) / (M_PI * diff)
        h[k] *= 0.54 - 0.46 * cos(2.0 * M_PI * k / (taps - 1))

    # Generate signal
    for i in range(n):
        s[i] = sin(i * 0.01) + sin(i * 0.005)

    # Filter and downsample
    total = 0.0
    for i in range(mid, n - mid, factor):
        acc = 0.0
        for k in range(taps):
            acc += h[k] * s[i - mid + k]
        total += acc

    free(h)
    free(s)
    return total
