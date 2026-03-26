# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute excess kurtosis and skewness of a deterministic dataset (Cython-optimized).

Keywords: statistics, kurtosis, skewness, moments, distribution, cython, benchmark
"""

from libc.math cimport sin, sqrt
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000000,))
def kurtosis(int n):
    """Compute excess kurtosis, skewness, and mean of a deterministic dataset."""
    cdef int i
    cdef double total = 0.0, mean, val, d, d2
    cdef double m2 = 0.0, m3 = 0.0, m4 = 0.0
    cdef double excess_kurt, skew

    # Pass 1: compute mean
    for i in range(n):
        total += sin(i * 0.01) * 50.0 + (i * 13 + 7) % 97
    mean = total / n

    # Pass 2: compute central moments
    for i in range(n):
        val = sin(i * 0.01) * 50.0 + (i * 13 + 7) % 97
        d = val - mean
        d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2

    m2 /= n
    m3 /= n
    m4 /= n

    if m2 == 0.0:
        return (0.0, 0.0, mean)

    excess_kurt = m4 / (m2 * m2) - 3.0
    skew = m3 / (m2 * sqrt(m2))

    return (excess_kurt, skew, mean)
