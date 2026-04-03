# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum of Poisson PMF values using iterative recurrence to avoid factorial overflow (Cython-optimized).

For lambda=5.0, compute PMF(x) for x=0..n-1 using pmf[x] = pmf[x-1] * k / x,
starting from pmf[0] = exp(-k). Track total sum, argmax, and midpoint PMF value.

Keywords: statistics, poisson, pmf, probability, iterative, cython, benchmark
"""

from libc.math cimport exp
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def poisson_pmf_sum(int n):
    """Compute sum of Poisson PMF values for x=0..n-1."""
    cdef double k = 5.0
    cdef int half_n = n // 2
    cdef double pmf, total_sum, max_pmf, pmf_at_half
    cdef int max_pmf_x, x

    # pmf[0] = exp(-k)
    pmf = exp(-k)
    total_sum = pmf
    max_pmf = pmf
    max_pmf_x = 0
    pmf_at_half = 0.0

    if half_n == 0:
        pmf_at_half = pmf

    with nogil:
        for x in range(1, n):
            pmf = pmf * k / x
            total_sum = total_sum + pmf
            if pmf > max_pmf:
                max_pmf = pmf
                max_pmf_x = x
            if x == half_n:
                pmf_at_half = pmf

    return (total_sum, max_pmf_x, pmf_at_half)
