# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Arithmetic coding interval width computation (Cython-optimized).

Keywords: compression, arithmetic coding, entropy, frequency, cython, benchmark
"""

from libc.math cimport log
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def arithmetic_coding_freq(int n):
    """Compute sum of log interval widths for arithmetic coding of n symbols."""
    cdef int i, sym
    cdef int freq[4]
    cdef double log_width_sum, log_width
    cdef double ratio[4]

    # Count frequencies
    freq[0] = 0; freq[1] = 0; freq[2] = 0; freq[3] = 0
    for i in range(n):
        sym = (i * 7 + 3) % 4
        freq[sym] += 1

    # Precompute log(freq[sym]/n) for each symbol
    for i in range(4):
        ratio[i] = log(<double>freq[i] / <double>n)

    # Simulate arithmetic coding
    log_width_sum = 0.0
    log_width = 0.0

    for i in range(n):
        sym = (i * 7 + 3) % 4
        log_width += ratio[sym]

        if (i + 1) % 100 == 0:
            log_width_sum += log_width
            log_width = 0.0

    log_width_sum += log_width

    return log_width_sum
