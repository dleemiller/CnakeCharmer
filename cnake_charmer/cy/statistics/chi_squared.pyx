# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute chi-squared statistic for n bins (Cython-optimized).

Observed values: o[i] = (i*7+3) % 50 + 1. Expected value: mean of observed.
Returns the chi-squared test statistic.

Keywords: statistics, chi-squared, hypothesis testing, goodness of fit, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def chi_squared(int n):
    """Compute chi-squared statistic for n bins."""
    cdef int i, obs
    cdef double obs_sum, expected, diff, chi2

    obs_sum = 0.0
    for i in range(n):
        obs = (i * 7 + 3) % 50 + 1
        obs_sum += obs

    expected = obs_sum / n

    chi2 = 0.0
    for i in range(n):
        obs = (i * 7 + 3) % 50 + 1
        diff = obs - expected
        chi2 += diff * diff / expected

    return chi2
