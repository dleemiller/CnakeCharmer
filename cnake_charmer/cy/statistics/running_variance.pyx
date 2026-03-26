# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Running variance using Welford's online algorithm (Cython-optimized).

Keywords: statistics, variance, welford, online, streaming, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def running_variance(int n):
    """Compute running variance using Welford's online algorithm."""
    cdef double mean = 0.0
    cdef double m2 = 0.0
    cdef double val, delta, delta2
    cdef int i

    for i in range(n):
        val = ((i * 17 + 5) % 1000) / 10.0
        delta = val - mean
        mean += delta / (i + 1)
        delta2 = val - mean
        m2 += delta * delta2

    if n < 2:
        return 0.0
    return m2 / n
