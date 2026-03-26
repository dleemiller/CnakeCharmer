# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute spectral centroid of a signal using DFT magnitudes (Cython-optimized).

The spectral centroid is the weighted mean of frequencies, weighted by
their magnitudes. Returns the centroid as a frequency bin index.

Keywords: dsp, spectral, centroid, DFT, frequency, magnitude, cython, benchmark
"""

from libc.math cimport sin, cos, sqrt, M_PI
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))
def spectral_centroid(int n):
    """Compute spectral centroid and return as frequency bin float."""
    cdef int i, k
    cdef int half_n = n // 2
    cdef double two_pi_over_n = 2.0 * M_PI / n
    cdef double re, im, mag, angle
    cdef double weighted_sum = 0.0
    cdef double magnitude_sum = 0.0
    cdef double *s = <double *>malloc(n * sizeof(double))
    if not s:
        raise MemoryError()

    # Pre-compute signal
    for i in range(n):
        s[i] = sin(i * 0.1) + 0.3 * sin(i * 0.3)

    for k in range(half_n + 1):
        re = 0.0
        im = 0.0
        for i in range(n):
            angle = two_pi_over_n * k * i
            re += s[i] * cos(angle)
            im -= s[i] * sin(angle)
        mag = sqrt(re * re + im * im)
        weighted_sum += k * mag
        magnitude_sum += mag

    free(s)

    if magnitude_sum == 0.0:
        return 0.0
    return weighted_sum / magnitude_sum
