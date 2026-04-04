# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply a second-order IIR biquad filter to a sinusoidal signal (Cython-optimized).

Uses direct form I with fixed coefficients. Returns the sum of the
filtered output signal.

Keywords: dsp, IIR, biquad, filter, recursive, signal, cython, benchmark
"""

from libc.math cimport sin
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def iir_biquad(int n):
    """Apply IIR biquad filter and return sum of filtered signal."""
    cdef int i
    cdef double b0 = 0.1, b1 = 0.2, b2 = 0.1
    cdef double a1 = -0.8, a2 = 0.2
    cdef double x0, x1 = 0.0, x2 = 0.0
    cdef double y0, y1 = 0.0, y2 = 0.0
    cdef double total = 0.0

    for i in range(n):
        x0 = sin(i * 0.05)
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        total += y0
        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0

    return total
