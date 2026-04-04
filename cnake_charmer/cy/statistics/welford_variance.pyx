# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Welford's online algorithm for population variance (Cython-optimized)."""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def welford_variance(int n):
    """Compute mean and population variance using Welford's online algorithm.

    Generates n values via a linear congruential generator:
      x[i] = ((i * 1664525 + 1013904223) & 0xFFFFFFFF) / 4294967296.0

    Args:
        n: Number of values.

    Returns:
        (mean, variance) as floats.
    """
    cdef unsigned int i, val
    cdef int count
    cdef double x, delta, mean, M2, variance

    mean = 0.0
    M2 = 0.0

    with nogil:
        for count in range(n):
            i = <unsigned int>count
            val = i * <unsigned int>1664525 + <unsigned int>1013904223
            x = <double>val / 4294967296.0
            delta = x - mean
            mean += delta / (count + 1)
            M2 += delta * (x - mean)

    variance = M2 / n
    return (mean, variance)
