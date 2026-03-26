# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Goertzel algorithm to detect a specific frequency bin (Cython-optimized).

Computes the power at a target frequency using the Goertzel recurrence,
which is more efficient than a full DFT for single-bin detection.

Keywords: dsp, Goertzel, frequency, detection, DFT, power, cython, benchmark
"""

from libc.math cimport sin, cos, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def goertzel(int n):
    """Compute power at target frequency using Goertzel algorithm."""
    cdef int i
    cdef int target_bin = 100
    cdef double two_pi = 2.0 * M_PI
    cdef double coeff = 2.0 * cos(two_pi * target_bin / n)
    cdef double s0 = 0.0, s1 = 0.0, s2 = 0.0
    cdef double sample, power

    for i in range(n):
        sample = sin(two_pi * i * 100 / n) + 0.5 * sin(two_pi * i * 300 / n)
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0

    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return power
