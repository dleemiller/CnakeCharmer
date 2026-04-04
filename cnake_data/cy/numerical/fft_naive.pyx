# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Naive DFT (Discrete Fourier Transform) of n complex values (Cython-optimized).

Computes the DFT using the O(n^2) direct summation formula and returns the
sum of magnitudes of all frequency bins.

Keywords: numerical, DFT, Fourier, transform, frequency, magnitude, cython, benchmark
"""

from libc.math cimport sin, cos, sqrt, M_PI
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def fft_naive(int n):
    """Compute naive DFT of n complex values and return sum of magnitudes."""
    cdef int j, k
    cdef double re, im, angle, total, two_pi_over_n
    cdef double *real_in = <double *>malloc(n * sizeof(double))
    if not real_in:
        raise MemoryError()

    for j in range(n):
        real_in[j] = sin(j * 0.1)

    total = 0.0
    two_pi_over_n = 2.0 * M_PI / n

    for k in range(n):
        re = 0.0
        im = 0.0
        for j in range(n):
            angle = two_pi_over_n * k * j
            re += real_in[j] * cos(angle)
            im -= real_in[j] * sin(angle)
        total += sqrt(re * re + im * im)

    free(real_in)
    return total
