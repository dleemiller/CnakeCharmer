# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Apply Hanning, Hamming, and Blackman windows to a signal (Cython-optimized).

Computes all three windowed signals and returns the sum of all values.

Keywords: dsp, window, Hanning, Hamming, Blackman, signal, cython, benchmark
"""

from libc.math cimport sin, cos, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def window_functions(int n):
    """Apply three window functions and return sum of all windowed values."""
    cdef int i
    cdef double sig, cos_val, cos_val2
    cdef double hanning, hamming, blackman
    cdef double two_pi_over_nm1 = 2.0 * M_PI / (n - 1)
    cdef double four_pi_over_nm1 = 4.0 * M_PI / (n - 1)
    cdef double total = 0.0

    for i in range(n):
        sig = sin(i * 0.1)
        cos_val = cos(two_pi_over_nm1 * i)
        cos_val2 = cos(four_pi_over_nm1 * i)

        hanning = 0.5 * (1.0 - cos_val)
        hamming = 0.54 - 0.46 * cos_val
        blackman = 0.42 - 0.5 * cos_val + 0.08 * cos_val2

        total += sig * hanning + sig * hamming + sig * blackman

    return total
