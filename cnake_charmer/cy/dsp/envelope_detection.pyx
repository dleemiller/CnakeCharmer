# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Hilbert transform envelope detection (Cython-optimized).

Uses a naive DFT-based approach to compute the analytic signal magnitude
and returns the sum of the envelope.

Keywords: dsp, envelope, Hilbert, transform, analytic, signal, cython, benchmark
"""

from libc.math cimport sin, cos, sqrt, M_PI
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def envelope_detection(int n):
    """Compute envelope via naive Hilbert transform and return sum."""
    cdef int i, k
    cdef int half_n = n // 2
    cdef double two_pi_over_n = 2.0 * M_PI / n
    cdef double quad, envelope, total, angle_factor
    cdef double *s = <double *>malloc(n * sizeof(double))
    if not s:
        raise MemoryError()

    for i in range(n):
        s[i] = sin(i * 0.1) * (1.0 + 0.5 * cos(i * 0.001))

    total = 0.0
    for i in range(n):
        quad = 0.0
        for k in range(1, half_n):
            quad += s[k] * sin(two_pi_over_n * k * i) * 2.0 / n
        for k in range(half_n + 1, n):
            quad -= s[k] * sin(two_pi_over_n * k * i) * 2.0 / n

        envelope = sqrt(s[i] * s[i] + quad * quad)
        total += envelope

    free(s)
    return total
