# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count zero crossings in a noisy signal (Cython-optimized).

Returns the total number of sign changes in the signal.

Keywords: dsp, zero-crossing, rate, sign, signal, cython, benchmark
"""

from libc.math cimport sin, cos
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def zero_crossing_rate(int n):
    """Count zero crossings in a signal."""
    cdef int i, count
    cdef double prev, curr

    count = 0
    prev = sin(0.0) * cos(0.0) + 0.1 * ((0 * 7 + 3) % 100 - 50) / 50.0

    for i in range(1, n):
        curr = sin(i * 0.1) * cos(i * 0.03) + 0.1 * ((i * 7 + 3) % 100 - 50) / 50.0
        if (prev >= 0.0 and curr < 0.0) or (prev < 0.0 and curr >= 0.0):
            count += 1
        prev = curr

    return count
