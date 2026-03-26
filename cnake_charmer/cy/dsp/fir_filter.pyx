# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply a 31-tap low-pass FIR filter to a sinusoidal signal (Cython-optimized).

Coefficients use a sinc-Hamming window design. Returns the sum of the
filtered output signal.

Keywords: dsp, FIR, filter, convolution, low-pass, Hamming, signal, cython, benchmark
"""

from libc.math cimport sin, cos, M_PI
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def fir_filter(int n):
    """Apply a 31-tap FIR filter and return sum of filtered signal."""
    cdef int taps = 31
    cdef int mid = 15
    cdef int i, k
    cdef double acc, total, diff
    cdef double *h = <double *>malloc(taps * sizeof(double))
    cdef double *s = <double *>malloc(n * sizeof(double))
    if not h or not s:
        raise MemoryError()

    # Design FIR coefficients: sinc * Hamming
    for k in range(taps):
        if k == mid:
            h[k] = 0.2
        else:
            diff = k - mid
            h[k] = sin(0.2 * M_PI * diff) / (M_PI * diff)
        h[k] *= 0.54 - 0.46 * cos(2.0 * M_PI * k / (taps - 1))

    # Generate signal
    for i in range(n):
        s[i] = sin(i * 0.01) + 0.5 * sin(i * 0.1)

    # Apply FIR filter
    total = 0.0
    for i in range(mid, n - mid):
        acc = 0.0
        for k in range(taps):
            acc += h[k] * s[i - mid + k]
        total += acc

    free(h)
    free(s)
    return total
