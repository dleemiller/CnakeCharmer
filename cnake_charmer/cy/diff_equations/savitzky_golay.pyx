# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply Savitzky-Golay filter (polynomial order 2, window 7) to a signal (Cython-optimized).

Signal is v[i] = sin(i*0.01) + 0.1*((i*7+3)%100-50)/50.0. Returns sum of smoothed values.

Keywords: signal processing, Savitzky-Golay, smoothing, convolution, filter, cython
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def savitzky_golay(int n):
    """Apply Savitzky-Golay filter to a noisy signal and return sum of smoothed values."""
    cdef int i, j
    cdef double total, val
    cdef double *signal = <double *>malloc(n * sizeof(double))
    if not signal:
        raise MemoryError()

    # SG coefficients for order 2, window 7: [-2, 3, 6, 7, 6, 3, -2] / 21
    cdef double coeffs[7]
    coeffs[0] = -2.0 / 21.0
    coeffs[1] = 3.0 / 21.0
    coeffs[2] = 6.0 / 21.0
    coeffs[3] = 7.0 / 21.0
    coeffs[4] = 6.0 / 21.0
    coeffs[5] = 3.0 / 21.0
    coeffs[6] = -2.0 / 21.0

    cdef int half_w = 3

    # Generate signal
    for i in range(n):
        signal[i] = sin(i * 0.01) + 0.1 * ((i * 7 + 3) % 100 - 50) / 50.0

    # Apply filter
    total = 0.0
    for i in range(half_w, n - half_w):
        val = 0.0
        for j in range(7):
            val += coeffs[j] * signal[i - half_w + j]
        total += val

    # Add unfiltered edge values
    for i in range(half_w):
        total += signal[i]
    for i in range(n - half_w, n):
        total += signal[i]

    free(signal)
    return total
