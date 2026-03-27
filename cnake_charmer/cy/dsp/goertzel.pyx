# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Goertzel algorithm to detect a specific frequency bin (Cython-optimized).

Computes the magnitude, phase cosine, and power at a target frequency using
the Goertzel recurrence.

Keywords: dsp, Goertzel, frequency, detection, DFT, power, cython, benchmark
"""

from libc.math cimport sin, cos, sqrt, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def goertzel(int n):
    """Compute DFT metrics at target frequency using Goertzel algorithm."""
    cdef int i
    cdef int target_bin = 100
    cdef double two_pi = 2.0 * M_PI
    cdef double coeff = 2.0 * cos(two_pi * target_bin / n)
    cdef double s0 = 0.0, s1 = 0.0, s2 = 0.0
    cdef double sample, power, w, real_part, imag_part, magnitude, phase_cos

    for i in range(n):
        sample = sin(two_pi * i * 100 / n) + 0.5 * sin(two_pi * i * 300 / n)
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0

    power = s1 * s1 + s2 * s2 - coeff * s1 * s2

    # Real and imaginary parts of DFT at target bin
    w = two_pi * target_bin / n
    real_part = s1 - s2 * cos(w)
    imag_part = s2 * sin(w)

    magnitude = sqrt(real_part * real_part + imag_part * imag_part)
    if magnitude > 0.0:
        phase_cos = real_part / magnitude
    else:
        phase_cos = 0.0

    return (magnitude, phase_cos, power)
